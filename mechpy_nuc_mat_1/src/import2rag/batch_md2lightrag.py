#!/usr/bin/env python3
"""
批量将 /home/zyx/A_project/mechpy/data/processed/markdown 目录下的markdown文件导入到RAG
专门用于处理已转换的学术论文markdown文件
"""

import asyncio
import os
import time
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_md2RAG.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class BatchRAGImporter:
    """批量RAG导入器"""
    
    def __init__(self):
        self.working_dir = '/home/zyx/A_project/mechpy/myKG'
        self.llm_model = 'qwen2.5:1.5b-instruct-q4_K_S'  # 使用小模型提高速度
        self.embed_model = 'bge-m3:latest'
        self.ollama_host = 'http://127.0.0.1:11434'
        self.rag = None
        self.processed_files = set()
        self.failed_files = []
        
    async def initialize(self):
        """初始化RAG"""
        try:
            logger.info('🔄 正在初始化RAG...')
            
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=ollama_model_complete,
                llm_model_name=self.llm_model,
                llm_model_kwargs={
                    'host': self.ollama_host,
                    'options': {'num_ctx': 4096},
                    'timeout': 600,  # 10分钟超时
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=lambda texts: ollama_embed(
                        texts=texts, embed_model=self.embed_model, host=self.ollama_host
                    ),
                ),
                kv_storage='JsonKVStorage',
                graph_storage='NetworkXStorage',
                vector_storage='NanoVectorDBStorage',
            )

            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            
            logger.info('✅ RAG初始化成功')
            return True
            
        except Exception as e:
            logger.error(f'❌ RAG初始化失败: {e}')
            return False
    
    def scan_markdown_files(self, base_dir):
        """扫描指定目录下的所有markdown文件"""
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.error(f'❌ 目录不存在: {base_path}')
            return []
        
        # 查找所有子目录中的.md文件
        md_files = []
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                # 在每个子目录中查找.md文件
                subdir_files = list(subdir.glob('*.md'))
                md_files.extend(subdir_files)
                logger.info(f'📁 在 {subdir.name} 中找到 {len(subdir_files)} 个markdown文件')
        
        logger.info(f'📄 总共找到 {len(md_files)} 个markdown文件')
        return md_files
    
    async def process_markdown_file(self, file_path):
        """处理单个markdown文件"""
        try:
            file_path = Path(file_path)
            
            # 检查文件是否已处理
            if str(file_path) in self.processed_files:
                logger.debug(f'⏭️  跳过已处理文件: {file_path.name}')
                return True
            
            # 检查文件是否有效
            if not file_path.exists():
                logger.warning(f'⚠️  文件不存在: {file_path}')
                self.failed_files.append((str(file_path), "文件不存在"))
                return False
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查内容是否有效
            if len(content.strip()) < 100:
                logger.warning(f'⚠️  文件内容过短: {file_path.name}')
                self.failed_files.append((str(file_path), "内容过短"))
                return False
            
            # 限制内容长度以避免超时
            content = content[:8000]  # 处理前8000个字符
            
            # 插入到RAG
            await self.rag.ainsert(content)
            
            # 标记为已处理
            self.processed_files.add(str(file_path))
            
            logger.info(f'✅ 成功导入: {file_path.name}')
            return True
            
        except Exception as e:
            error_msg = f'处理失败: {str(e)}'
            logger.error(f'❌ 处理文件失败 {file_path}: {e}')
            self.failed_files.append((str(file_path), error_msg))
            return False
    
    async def batch_process(self, base_dir, batch_size=10, delay=2):
        """批量处理所有markdown文件"""
        md_files = self.scan_markdown_files(base_dir)
        
        if not md_files:
            logger.warning('⚠️  未找到任何markdown文件')
            return 0, []
        
        logger.info(f'🚀 开始批量处理 {len(md_files)} 个文件...')
        
        processed_count = 0
        total_files = len(md_files)
        
        # 使用进度条显示处理进度
        with tqdm(total=total_files, desc="导入进度", unit="file") as pbar:
            for i in range(0, total_files, batch_size):
                batch = md_files[i:i+batch_size]
                
                # 处理当前批次
                batch_tasks = []
                for file_path in batch:
                    task = self.process_markdown_file(file_path)
                    batch_tasks.append(task)
                
                # 等待当前批次完成
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 统计成功数量
                batch_success = sum(1 for result in batch_results if result is True)
                processed_count += batch_success
                
                # 更新进度条
                pbar.update(len(batch))
                
                # 显示批次统计
                logger.info(f'📊 批次 {i//batch_size + 1}: {batch_success}/{len(batch)} 成功')
                
                # 批次间延迟，避免过载
                if i + batch_size < total_files:
                    logger.info(f'⏳ 等待 {delay} 秒后继续...')
                    await asyncio.sleep(delay)
        
        # 输出最终统计
        logger.info(f'🎉 批量处理完成!')
        logger.info(f'📊 成功导入: {processed_count}/{total_files} 文件')
        logger.info(f'❌ 失败文件: {len(self.failed_files)} 个')
        
        if self.failed_files:
            logger.info('📋 失败文件列表:')
            for file_path, error in self.failed_files:
                logger.info(f'   - {Path(file_path).name}: {error}')
        
        return processed_count, self.failed_files

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量导入markdown文件到RAG')
    parser.add_argument('--base-dir', 
                       default='/home/zyx/A_project/mechpy/data/processed/markdown',
                       help='markdown文件基础目录路径')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='每批次处理的文件数量')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='批次间的延迟时间(秒)')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅扫描文件，不实际导入')
    
    args = parser.parse_args()
    
    print(f"""
    🚀 批量Markdown导入RAG工具
    ================================
    
    配置信息:
    📁 基础目录: {args.base_dir}
    📦 批次大小: {args.batch_size}
    ⏱️  批次延迟: {args.delay}秒
    🧪 试运行: {'是' if args.dry_run else '否'}
    
    日志文件: batch_md2RAG.log
    """)
    
    # 创建导入器
    importer = BatchRAGImporter()
    
    # 初始化RAG
    if not args.dry_run:
        if not await importer.initialize():
            logger.error('❌ RAG初始化失败，程序退出')
            return
    
    # 扫描文件
    md_files = importer.scan_markdown_files(args.base_dir)
    
    if not md_files:
        logger.warning('⚠️  未找到任何markdown文件，程序退出')
        return
    
    if args.dry_run:
        logger.info('🧪 试运行模式，仅扫描文件，不进行导入')
        logger.info(f'📄 找到 {len(md_files)} 个markdown文件:')
        for file_path in md_files[:10]:  # 只显示前10个
            logger.info(f'   - {file_path.name}')
        if len(md_files) > 10:
            logger.info(f'   ... 还有 {len(md_files) - 10} 个文件')
        return
    
    # 执行批量导入
    start_time = time.time()
    processed_count, failed_files = await importer.batch_process(
        args.base_dir, 
        batch_size=args.batch_size, 
        delay=args.delay
    )
    end_time = time.time()
    
    # 输出总结报告
    elapsed_time = end_time - start_time
    logger.info(f"""
    📊 导入总结报告
    ================
    📁 基础目录: {args.base_dir}
    📄 总文件数: {len(md_files)}
    ✅ 成功导入: {processed_count}
    ❌ 导入失败: {len(failed_files)}
    ⏱️  总耗时: {elapsed_time:.2f}秒
    📈 平均速度: {len(md_files)/elapsed_time:.2f} 文件/秒
    """)
    
    # 保存失败文件列表
    if failed_files:
        failed_file_path = 'batch_md2RAG_failed.txt'
        with open(failed_file_path, 'w', encoding='utf-8') as f:
            f.write("失败文件列表:\n")
            for file_path, error in failed_files:
                f.write(f"{file_path}: {error}\n")
        logger.info(f'📋 失败文件列表已保存到: {failed_file_path}')

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('🛑 用户中断程序执行')
    except Exception as e:
        logger.error(f'💥 程序执行出错: {e}')
        raise
