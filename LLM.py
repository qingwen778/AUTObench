import time
from openai import OpenAI
import requests
class openrouter_api:
    def __init__(self, model="deepseek/deepseek-v3.2"):
        """
        初始化 OpenRouter API
        """
        self.client = OpenAI(
            api_key="",
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        print("model:",model)

    def chat(self, prompt, max_tokens=5000, temperature=0.7, return_usage=False):

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            content = response.choices[0].message.content
            
            if return_usage:
                # 返回内容和token使用信息
                usage_info = {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                }
                return {
                    'content': content,
                    'usage': usage_info
                }
            else:
                return content

        except Exception as e:
            print("Error:", e)
            return None


class siliconflow:
    def __init__(self, api_key="", api_url="https://api.siliconflow.cn/v1/chat/completions", model="deepseek/deepseek-v3.2"):

        self.api_key = api_key
        self.api_url = api_url
        self.model = model

    def chat(self, prompt, max_tokens=5000, temperature=0.7, return_usage=False):

        payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "请按照我的要求生成"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1.3,
                "max_tokens": 8000
            }

        headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # 调用 API
        response = requests.post(self.api_url, json=payload, headers=headers)

        if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                return result
        else:
                return None

class deepseek_official:
    def __init__(self, api_key="", model="deepseek-chat"):
        """
        初始化 DeepSeek 官方 API
        api_key: DeepSeek API密钥
        model: 模型名称，默认为 deepseek-chat
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    def chat(self, prompt, max_tokens=5000, temperature=0.7, return_usage=False):

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            content = response.choices[0].message.content
            
            if return_usage:
                # 返回内容和token使用信息
                usage_info = {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                }
                return {
                    'content': content,
                    'usage': usage_info
                }
            else:
                return content

        except Exception as e:
            print("Error:", e)
            return None




if __name__ == "__main__":


    llm = openrouter_api(model="deepseek/deepseek-v3.2")

    print("=== 普通调用 ===")
    res = llm.chat("给我解释一下 LoRA 微调。")
    print(res)

