"""
This is an example of connecting to the Postgres database.
Currently the dockerfile runs the DB but does NOT access GPU resources.
Stuff like cudotoolkit needs to be added to the image.
"""

import psycopg2
import os
import re
import json
from vllm import LLM, SamplingParams
from typing import Literal, TypedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

POSTGRES_CONN_INFO = os.getenv(
    "POSTGRES_CONN_INFO",
    "dbname=msft_customers user=julia password=123 host=localhost port=5432",
)

model = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"

Role = Literal["system", "user", "assistant"]


def get_schema(conn_info):
    conn = psycopg2.connect(conn_info)
    cur = conn.cursor()

    # Get user tables (ignore system schemas)
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [row[0] for row in cur.fetchall()]

    schema = ""
    for table in tables:
        cur.execute(
            f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """,
            (table,),
        )
        columns = cur.fetchall()

        schema += f"Table: {table}\n  Columns:\n"
        for name, dtype, is_nullable, default in columns:
            notnull = "NOT NULL" if is_nullable == "NO" else ""
            default_str = f"DEFAULT {default}" if default else ""
            schema += f"    {name} ({dtype}) {notnull} {default_str}\n"

        # Foreign keys info
        cur.execute(
            f"""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
        """,
            (table,),
        )
        fks = cur.fetchall()
        if fks:
            schema += "  Foreign Keys:\n"
            for from_col, ref_table, to_col in fks:
                schema += f"    {from_col} → {ref_table}({to_col})\n"

        schema += "\n"

    cur.close()
    conn.close()
    return schema


def execute_sql(conn_info, query):
    conn = psycopg2.connect(conn_info)
    cur = conn.cursor()
    cur.execute(query)
    try:
        rows = cur.fetchall()
    except psycopg2.ProgrammingError:
        # No results to fetch (e.g., for INSERT)
        rows = []
    conn.commit()
    cur.close()
    conn.close()
    return rows


class Message(TypedDict):
    role: Role
    content: str


class ChatState:
    def __init__(self):
        self.conversation: list[Message] = []

    def append(self, role: Role, content: str):
        self.conversation.append({"role": role, "content": content})

    def modify_message(self, role: Role, content: str):
        for i in range(len(self.conversation) - 1, -1, -1):
            if self.conversation[i]["role"] == role:
                self.conversation[i]["content"] += content
                break

    def delete_last(self):
        if self.conversation:
            self.conversation.pop()


class OfflineLLM:
    def __init__(self, model_name: str = model):
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        parallel_size = len(cuda_visible.split(",")) if cuda_visible else 1
        self.llm = LLM(model=model_name, tensor_parallel_size=parallel_size)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        )

    def complete(
        self,
        history: list[Message],
        system_prompt: str = "",
    ) -> str:
        conversation = [{"role": "system", "content": system_prompt}] + history

        print("TEST conversation", conversation)

        # using chat instead of generate so that vLLM will automatically format
        #   the messages according to the model's expected chat template
        outputs = self.llm.chat(conversation, self.sampling_params)
        return outputs[0].outputs[0].text


schema = get_schema(POSTGRES_CONN_INFO)


# user_email = "katherine0@adventure-works.com"
user_id = 29485
# user_first_name = "Katherine"
# user_last_name = "Harding"

sql_system_prompt = (
    "You are a helpful assistant that generates SQL queries for a customer database with the following schema:\n"
    f"{schema}\n"
    "The user you are helping is logged in and is only allowed to access their own data. Never generate queries that expose other users’ data, e.g. accessing other users' orders"
    " The current user is represented by:"
    # f"Email: {user_email} "
    f"CustomerID: {user_id} "
    # f"User Name: {user_first_name} {user_last_name} "
    "Do not generate any INSERT, UPDATE, DELETE, or DROP statements."
    "Return the SQL query in JSON format with key 'query'."
    "If the user's intent is unclear or seems dangerous, return a query that returns 'ERROR: dangerous query'"
)

text_system_prompt = "You are a helpful assistant that answer's user requests based on SQL queries for a Microsoft customer database."


class Agent:
    def __init__(self, conn_info):
        self.conn_info = conn_info
        self.schema = get_schema(conn_info)
        self.llm = OfflineLLM()
        self.state = ChatState()

    def run_turn(self, user_input):
        self.state.append("user", user_input)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            response = self.llm.complete(
                self.state.conversation, system_prompt=sql_system_prompt
            )
            print(f"TEST query generation llm response: {response}")

            match = re.search(r'\{[^{}]*"query"\s*:\s*"([^"]*?)"\s*\}', response)

            if match:
                json_str = match.group(0)
                query = json.loads(json_str)["query"]
                print(f"TEST SQL query: {query}")

                try:
                    result = execute_sql(self.conn_info, query)
                    print(f"TEST SQL query result: {result}")

                    query_result_string = f"Using the following query and results\n query: {query}\n result: {result}"
                    self.state.append("user", query_result_string)

                    break

                except Exception as e:
                    retry_count += 1
                    error_message = f"SQL error: {e}"
                    print(f"SQL error on attempt {retry_count}: {error_message}")

                    if retry_count < max_retries:
                        error_feedback = f"The previous query failed with error: {error_message}. Please generate a corrected SQL query that fixes this issue. Remember to follow the database schema: {self.schema}"
                        self.state.append("assistant", response)
                        self.state.append("user", error_feedback)
                    else:
                        return f"Failed to generate valid SQL query after {max_retries} attempts. Last error: {error_message}"

            else:
                retry_count += 1
                error_message = "No valid JSON with 'query' found in response."
                print(f"Parse error on attempt {retry_count}: {error_message}")

                if retry_count < max_retries:
                    error_feedback = f'The previous response did not contain a valid JSON with \'query\' field. Please respond with a valid JSON format containing the SQL query like: {{"query": "SELECT * FROM table_name"}}'
                    self.state.append("assistant", response)
                    self.state.append("user", error_feedback)
                else:
                    return f"Failed to generate valid query format after {max_retries} attempts."

        response = self.llm.complete(
            self.state.conversation, system_prompt=text_system_prompt
        )
        self.state.delete_last()
        self.state.append("assistant", response)
        return response


# agent = Agent(POSTGRES_CONN_INFO)

if __name__ == "__main__":
    print("Schema:", schema)
    # while True:
    #     try:
    #         user_input = input("\nUser: ")
    #         if user_input.lower() in ["exit", "quit"]:
    #             break

    #         reply = agent.run_turn(user_input)
    #         print("\n\nAssistant:", reply, "\n\n")

    #     except KeyboardInterrupt:
    #         break
