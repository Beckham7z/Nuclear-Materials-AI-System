import transformers

def calculate_token_usage(input_text, chat_tokenizer_dir="/mnt/d/sqdata/mechpy/llm/token_calc/DeepSeek"):
        """
        计算输入文本的 token 用量。

        参数:
        input_text (str): 要计算 token 用量的输入文本。
        chat_tokenizer_dir (str): Tokenizer 的预训练模型路径，默认为当前目录。

        返回:
        int: 输入文本的 token 数量。
        """
        # 加载 Tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
        chat_tokenizer_dir, trust_remote_code=True
        )

        # 编码输入文本
        encoded_input = tokenizer.encode(input_text)

        # 计算 token 数量
        token_count = len(encoded_input)

        return token_count

# 示例调用
def main():
        # 示例输入文本
        input_text = "Hello, how are you?"

        # 计算 token 用量
        token_usage = calculate_token_usage(input_text)

        # 打印结果
        print(f"Token用量: {token_usage}")
if __name__ == "__main__":
     main()