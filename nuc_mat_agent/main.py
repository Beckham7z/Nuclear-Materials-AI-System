"""
Nuclear Material Agent - 主程序入口
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuc_mat_agent.web_ui import main


if __name__ == "__main__":
    # 运行 Streamlit 应用
    # 命令: streamlit run nuc_mat_agent/main.py --server.port 8502
    main()
