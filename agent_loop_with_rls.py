
import sqlite3
file_path = "../data/msft_customers.db"


def get_schema(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall();
    schema = ""
    for table, in tables:
        if table in {"sqlite_sequence", "ErrorLog", "BuildVersion"}:
            # Skip irrelevant tables
            continue
        # if table.startswith("Product"):
        #     # ignore product tables for now
        #     continue
        
        # PRAGMA is a SQLite keyword used for reading or modifying database metadata
        cursor.execute(f"PRAGMA table_info({table});") 
        schema += f"Table: {table}\n  Columns:\n"
        for col in cursor.fetchall():
            name = col[1]
            dtype = col[2]
            pk_flag = "PRIMARY KEY" if col[5] else ""
            notnull = "NOT NULL" if col[3] else ""
            schema += f"    {name} ({dtype}) {notnull} {pk_flag}\n"
    
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()
        if fks:
            schema +="  Foreign Keys:"
            for fk in fks:
                id_, seq, ref_table, from_col, to_col, on_update, on_delete, match = fk
                schema +=f"    {from_col} → {ref_table}({to_col})\n"
        
        schema += "\n"


    conn.close()
    return schema

def execute_sql(db_path: str, query: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows


print(get_schema(file_path))


class SQLiteRLS:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.current_user_id = None
        
    def set_user_context(self, user_id: int):
        self.current_user_id = user_id
        
    def execute_with_rls(self, query: str, params: tuple = ()):
        """Automatically inject user filters into queries"""
        modified_query = self._inject_user_filter(query)
        return self.db.execute(modified_query, params)
    
    def _inject_user_filter(self, query: str) -> str:
        """Modify query to include user ownership checks"""
        # Parse the query to identify table and add user filter
        # This is a simplified example - use proper SQL parsing in production
        
        if query.strip().upper().startswith('SELECT'):
            # Extract table name (simplified)
            import re
            table_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
            if table_match:
                table = table_match.group(1)
                
                # Define which column represents ownership for each table
                ownership_columns = {
                    'users': 'user_id',
                    'addresses': 'user_id',
                    'orders': 'customer_id'
                }
                
                if table in ownership_columns:
                    ownership_col = ownership_columns[table]
                    
                    # Add WHERE clause or AND condition
                    if 'WHERE' in query.upper():
                        query += f" AND {ownership_col} = {self.current_user_id}"
                    else:
                        query += f" WHERE {ownership_col} = {self.current_user_id}"
        
        return query


from typing import Literal, TypedDict

Role = Literal["system", "user",  "assistant"] 

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


from vllm import LLM, SamplingParams
import os

class OfflineLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8"):
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        parallel_size = len(cuda_visible.split(",")) if cuda_visible else 1
        self.llm = LLM(model=model_name, tensor_parallel_size=parallel_size)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        )

    def complete(self, history: list[Message], system_prompt: str = "", ) -> str:
        conversation = [{
            "role": "system",
            "content": system_prompt
        }] + history

        print("TEST conversation", conversation)

        # using chat instead of generate so that vLLM will automatically format 
        #   the messages according to the model's expected chat template
        outputs = self.llm.chat(conversation, self.sampling_params)
        return outputs[0].outputs[0].text



# import re

# SENSITIVE_FIELDS = {"PasswordHash", "PasswordSalt", "CreditCardApprovalCode", "AccountNumber"}

# def detect_violation(query: str) -> str | None:
#     lowered = query.lower()

#     if any(keyword in lowered for keyword in ["insert", "update", "delete", "drop", "alter"]):
#         return "Write/query modification detected"

#     for field in SENSITIVE_FIELDS:
#         if field.lower() in lowered:
#             return f"Sensitive field accessed: {field}"

#     if "where customerid" not in lowered:
#         return "Possible multi-user data leak: no CustomerID restriction"

#     return None

# def detect_result_violation(result: list[tuple]) -> str | None:
#     result_str = str(result)
#     if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", result_str):
#         return "Email address leak"
#     if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", result_str):
#         return "Phone number leak"
#     return None



schema = get_schema(file_path)

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

text_system_prompt = (
    "You are a helpful assistant that answer's user requests based on SQL queries for a Microsoft customer database."
)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"


import re
response = '... some log ... {"query": "SELECT a.AddressLine1, a.AddressLine2, a.City, a.StateProvince, a.CountryRegion, a.PostalCode FROM Address a JOIN CustomerAddress ca ON a.AddressID = ca.AddressID WHERE ca.CustomerID = 11 AND ca.AddressType = \'Main Address\';"} ...more logs'
match = re.search(r'\{.*?"query"\s*:\s*".*?"\}', response)
match.group(0)


import json

class Agent:
    def __init__(self, db_path):
        self.db_path = db_path
        self.schema = get_schema(db_path)
        self.llm = OfflineLLM()
        self.state = ChatState()


    def run_turn(self, user_input):
        self.state.append("user", user_input)
        
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
        
            response = self.llm.complete(self.state.conversation, system_prompt=sql_system_prompt)
            print(f"TEST query generation llm response: {response}")

            match = re.search(r'\{[^{}]*"query"\s*:\s*"([^"]*?)"\s*\}', response)

            if match:
                json_str = match.group(0)
                query = json.loads(json_str)["query"]
                print(f"TEST SQL query: {query}")

                try:
                    result = rls.execute_with_rls(self.db_path, query)
                    print(f"TEST SQL query result: {result}")

                    query_result_string = f"Using the following query and results\n query: {query}\n result: {result}"
                    self.state.append("user", query_result_string)

                    # TODO detect violation

                    break

                except Exception as e:
                    # TODO detect violation

                    retry_count += 1
                    error_message = f"SQL error: {e}"
                    print(f"SQL error on attempt {retry_count}: {error_message}")

                    if retry_count < max_retries:
                        error_feedback = f"The previous query failed with error: {error_message}. Please generate a corrected SQL query that fixes this issue. Remember to follow the database schema: {self.schema}"
                        self.state.append("assistant", response)  # Add the failed response
                        self.state.append("user", error_feedback)  # Add error feedback for retry
                    else:
                        return f"Failed to generate valid SQL query after {max_retries} attempts. Last error: {error_message}"

                    

            else:
                retry_count += 1
                error_message = "No valid JSON with 'query' found in response."
                print(f"Parse error on attempt {retry_count}: {error_message}")
                
                if retry_count < max_retries:
                    error_feedback = f"The previous response did not contain a valid JSON with 'query' field. Please respond with a valid JSON format containing the SQL query like: {{\"query\": \"SELECT * FROM table_name\"}}"
                    self.state.append("assistant", response)  # Add the failed response
                    self.state.append("user", error_feedback)  # Add error feedback for retry
                else:
                    return f"Failed to generate valid query format after {max_retries} attempts."


    
        response = self.llm.complete(self.state.conversation, system_prompt=text_system_prompt)

        self.state.delete_last()  # Remove the last user input (query result string)
        self.state.append("assistant", response)
        return response


rls = SQLiteRLS(file_path)
rls.set_user_context(user_id)
agent = Agent(file_path)

while True:
    try:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        reply = agent.run_turn(user_input)
        print("\n\nAssistant:", reply, "\n\n")


    except KeyboardInterrupt:
        break


