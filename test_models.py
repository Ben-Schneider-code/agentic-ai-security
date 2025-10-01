#!/usr/bin/env python3
"""
vLLM Server Tester
Tests two vLLM OpenAI-compatible API endpoints using the OpenAI library
"""

import requests
import json
import time
import argparse
from typing import Dict, Any, Optional, List
from openai import OpenAI

class VLLMTester:
    def __init__(self, base_urls: List[str]):
        self.base_urls = base_urls
        self.session = requests.Session()  # Still needed for health checks
        self.clients = {}
        
        # Create OpenAI clients for each base URL
        for base_url in base_urls:
            self.clients[base_url] = OpenAI(
                api_key="fake-key",  # vLLM doesn't require real API key
                base_url=f"{base_url}/v1"
            )
    
    def check_health(self, base_url: str) -> bool:
        """Check if the server is healthy"""
        try:
            response = self.session.get(f"{base_url}/health", timeout=10)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Health check failed for {base_url}: {e}")
            return False
    
    def get_models(self, base_url: str) -> Optional[List[str]]:
        """Get available models from the server"""
        try:
            client = self.clients[base_url]
            models_response = client.models.list()
            return [model.id for model in models_response.data]
        except Exception as e:
            print(f"Failed to get models from {base_url}: {e}")
            return None
    
    def test_completion(self, base_url: str, model: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Test chat completion endpoint"""
        try:
            client = self.clients[base_url]
            print(f"Sending request to {base_url}...")
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7,
                stream=False
            )
            
            end_time = time.time()
            
            # Convert response to dict-like format for consistency
            result = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                "_response_time": end_time - start_time
            }
            return result
            
        except Exception as e:
            print(f"Completion request failed for {base_url}: {e}")
            return None
    
    def test_streaming_completion(self, base_url: str, model: str, prompt: str) -> bool:
        """Test streaming chat completion endpoint"""
        try:
            client = self.clients[base_url]
            print(f"Testing streaming from {base_url}...")
            
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7,
                stream=True
            )
            
            chunks_received = 0
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    chunks_received += 1
            
            print(f"Received {chunks_received} streaming chunks")
            return chunks_received > 0
            
        except Exception as e:
            print(f"Streaming request failed for {base_url}: {e}")
            return False
    
    def run_comprehensive_test(self, test_prompt: str = "What is the capital of France?"):
        """Run comprehensive tests on both servers"""
        print("="*60)
        print("Starting comprehensive vLLM server tests")
        print("="*60)
        
        for i, base_url in enumerate(self.base_urls, 1):
            print(f"\n--- Testing Server {i}: {base_url} ---")
            
            # Health check
            print("1. Health check...")
            if not self.check_health(base_url):
                print("❌ Server is not healthy, skipping other tests")
                continue
            print("✅ Server is healthy")
            
            # Get models
            print("2. Getting available models...")
            models = self.get_models(base_url)
            if not models:
                print("❌ Failed to get models")
                continue
                
            if not models:
                print("❌ No models available")
                continue
                
            model_name = models[0]
            print(f"✅ Found {len(models)} model(s), using: {model_name}")
            
            # Test completion
            print("3. Testing chat completion...")
            completion_result = self.test_completion(base_url, model_name, test_prompt)
            if completion_result:
                response_time = completion_result.get('_response_time', 0)
                content = completion_result['choices'][0]['message']['content']
                usage = completion_result.get('usage', {})
                tokens_used = usage.get('total_tokens', 'unknown') if usage else 'unknown'
                
                print(f"✅ Completion successful")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Tokens used: {tokens_used}")
                print(f"   Response: {content[:100]}...")
            else:
                print("❌ Completion failed")
            
            # Test streaming
            print("4. Testing streaming completion...")
            if self.test_streaming_completion(base_url, model_name, "Count from 1 to 5"):
                print("✅ Streaming works")
            else:
                print("❌ Streaming failed")
        
        print("\n" + "="*60)
        print("Testing complete!")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Test vLLM servers")
    parser.add_argument(
        "--urls", 
        nargs='+',
        default=["http://localhost:8000", "http://localhost:8001"],
        help="Base URLs of vLLM servers (default: http://localhost:8000 http://localhost:8001)"
    )
    parser.add_argument(
        "--prompt", 
        default="What is the capital of France?",
        help="Test prompt to send to servers"
    )
    
    args = parser.parse_args()
    
    tester = VLLMTester(args.urls)
    tester.run_comprehensive_test(args.prompt)

if __name__ == "__main__":
    main()