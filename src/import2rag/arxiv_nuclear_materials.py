#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
arXiv核材料文献爬虫 - 专门针对"nuclear materials"关键词
下载约200篇文献到F:\nuclear_material_spider\arxiv_pdfs
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import os
import logging
from urllib.parse import urlencode
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arxiv_nuclear_materials.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArXivNuclearMaterialsCrawler:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.data = []  # 存储所有文献数据
        self.download_dir = r"F:\nuclear_material_spider\arxiv_pdfs"
        
        # 注册命名空间
        self.namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        # 确保下载目录存在
        os.makedirs(self.download_dir, exist_ok=True)
        logger.info(f"下载目录: {self.download_dir}")

    def search_nuclear_materials(self, max_results=200):
        """搜索核材料相关文献"""
        logger.info(f"开始搜索核材料文献，目标数量: {max_results}篇")
        
        # 主要搜索关键词
        search_terms = ["nuclear materials"]
        
        # 计算每个关键词需要获取的文献数量
        per_term_max = max_results
        
        for term in search_terms:
            logger.info(f"搜索关键词: {term}")
            self._search_arxiv(term, per_term_max)
            time.sleep(2)  # 避免频繁请求

    def _search_arxiv(self, search_term, max_results=200):
        """调用arXiv API搜索文献"""
        # 由于arXiv API每次最多返回1000条结果，我们分批次获取
        batch_size = 100  # 每次请求的最大结果数
        start_index = 0
        total_fetched = 0
        
        while total_fetched < max_results:
            current_batch = min(batch_size, max_results - total_fetched)
            
            params = {
                'search_query': f'all:"{search_term}"',
                'start': start_index,
                'max_results': current_batch,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            try:
                url = f"{self.base_url}?{urlencode(params)}"
                logger.info(f"请求URL: {url}")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    batch_data = self._parse_arxiv_response(response.text, search_term)
                    total_fetched += len(batch_data)
                    logger.info(f"批次 {start_index//batch_size + 1}: 获取到 {len(batch_data)} 篇文献，总计 {total_fetched} 篇")
                    
                    # 如果本次获取的数据少于请求数量，说明没有更多数据了
                    if len(batch_data) < current_batch:
                        logger.info("已获取所有可用文献")
                        break
                        
                else:
                    logger.warning(f"arXiv API请求失败: 状态码 {response.status_code}")
                    break

            except Exception as e:
                logger.error(f"搜索arXiv出错: {str(e)}")
                break
            
            start_index += current_batch
            time.sleep(3)  # 批次间延迟

    def _parse_arxiv_response(self, xml_content, search_term):
        """解析XML响应"""
        batch_data = []
        try:
            root = ET.fromstring(xml_content)
            
            for entry in root.findall('atom:entry', self.namespaces):
                try:
                    # 提取基本信息
                    title = entry.find('atom:title', self.namespaces).text.strip()
                    summary = entry.find('atom:summary', self.namespaces).text.strip() if entry.find('atom:summary', self.namespaces) else ""
                    
                    # 处理作者列表
                    authors = []
                    for author in entry.findall('atom:author', self.namespaces):
                        author_name = author.find('atom:name', self.namespaces).text
                        if author_name:
                            authors.append(author_name)

                    # 出版日期和PDF链接
                    published = entry.find('atom:published', self.namespaces).text if entry.find('atom:published', self.namespaces) else ""
                    pdf_link = next((link.get('href') for link in entry.findall('atom:link', self.namespaces) if link.get('title') == 'pdf'), "")
                    arxiv_id = entry.find('atom:id', self.namespaces).text.split('/')[-1] if entry.find('atom:id', self.namespaces) else ""
                    
                    # 尝试获取DOI
                    doi = ""
                    doi_element = entry.find('arxiv:doi', self.namespaces)
                    if doi_element is not None and doi_element.text.strip():
                        doi = doi_element.text.strip()
                    
                    if not doi:
                        for link in entry.findall('atom:link', self.namespaces):
                            link_href = link.get('href', '')
                            if link.get('rel') == 'related' and 'doi.org' in link_href:
                                doi = link_href.split('doi.org/')[-1].strip()
                                break

                    # 组装文献数据
                    article_data = {
                        'arxiv_id': arxiv_id,
                        'doi': doi,
                        'title': title,
                        'authors': ', '.join(authors),
                        'abstract': summary,
                        'published_date': published,
                        'pdf_url': pdf_link,
                        'search_term': search_term,
                        'crawl_time': pd.Timestamp.now()
                    }

                    batch_data.append(article_data)
                    logger.info(f"提取文献: {title[:60]}...")

                except Exception as e:
                    logger.error(f"解析单篇文献出错: {str(e)}")
                    continue

            # 将批次数据添加到总数据中
            self.data.extend(batch_data)
            logger.info(f"解析完成: 获取到 {len(batch_data)} 篇文献")

        except Exception as e:
            logger.error(f"解析XML出错: {str(e)}")
        
        return batch_data

    def download_pdfs(self):
        """下载PDF文件"""
        if not self.data:
            logger.warning("无文献数据可下载")
            return

        downloaded_count = 0
        failed_count = 0

        for i, article in enumerate(self.data, 1):
            try:
                pdf_url = article.get('pdf_url', "")
                arxiv_id = article.get('arxiv_id', "")
                
                if not pdf_url or not arxiv_id:
                    logger.warning(f"跳过异常条目（无PDF链接或无arXiv ID）")
                    failed_count += 1
                    continue

                # 生成安全的文件名
                safe_filename = self._generate_safe_filename(arxiv_id, article.get('title', ''))
                filepath = os.path.join(self.download_dir, safe_filename)

                # 跳过已下载的文件
                if os.path.exists(filepath):
                    logger.info(f"文件已存在，跳过: {safe_filename}")
                    downloaded_count += 1
                    continue

                # 下载PDF
                logger.info(f"正在下载 [{i}/{len(self.data)}]: {safe_filename}")
                response = requests.get(pdf_url, timeout=60)
                
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    downloaded_count += 1
                    logger.info(f"下载成功: {safe_filename}")
                else:
                    logger.warning(f"下载失败（状态码{response.status_code}）: {safe_filename}")
                    failed_count += 1

                # 延迟控制，避免触发反爬
                time.sleep(2)

            except Exception as e:
                logger.error(f"下载文献 {article.get('arxiv_id', '未知ID')} 出错: {str(e)}")
                failed_count += 1
                continue

        logger.info(f"PDF下载完成: 成功 {downloaded_count} 篇 / 失败 {failed_count} 篇 / 总计 {len(self.data)} 篇")

    def _generate_safe_filename(self, arxiv_id, title):
        """生成安全的文件名"""
        # 清理标题中的非法字符
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
        safe_title = safe_title.replace(' ', '_')[:50]  # 限制长度
        
        # 组合文件名: arXiv ID + 标题前部分
        filename = f"{arxiv_id}_{safe_title}.pdf"
        return filename

    def save_to_excel(self, filename="nuclear_materials_literature.xlsx"):
        """保存文献元数据到Excel"""
        if not self.data:
            logger.warning("无文献数据可保存")
            return

        df = pd.DataFrame(self.data)
        
        # 调整列顺序
        column_order = ['arxiv_id', 'doi', 'title', 'authors', 'published_date', 'pdf_url', 'search_term', 'crawl_time']
        df = df[column_order]
        
        # 确保输出目录存在
        os.makedirs('output', exist_ok=True)
        output_path = f'output/{filename}'
        
        df.to_excel(output_path, index=False)
        logger.info(f"文献数据已保存到: {output_path}")
        return output_path

    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'total_articles': len(self.data),
            'articles_with_doi': len([article for article in self.data if article.get('doi')]),
            'unique_authors': len(set(author for article in self.data for author in article.get('authors', '').split(', '))),
            'earliest_date': min([article.get('published_date', '') for article in self.data if article.get('published_date')], default=''),
            'latest_date': max([article.get('published_date', '') for article in self.data if article.get('published_date')], default='')
        }
        return stats


def main():
    """主函数"""
    logger.info("=== arXiv核材料文献爬虫启动 ===")
    
    # 初始化爬虫
    crawler = ArXivNuclearMaterialsCrawler()
    
    try:
        # 1. 搜索核材料文献
        crawler.search_nuclear_materials(max_results=200)
        
        # 2. 显示统计信息
        stats = crawler.get_statistics()
        logger.info(f"文献统计: 总计 {stats['total_articles']} 篇，含DOI {stats['articles_with_doi']} 篇")
        
        # 3. 下载PDF文件
        crawler.download_pdfs()
        
        # 4. 保存元数据到Excel
        crawler.save_to_excel()
        
        logger.info("=== 爬虫任务完成 ===")
        
    except Exception as e:
        logger.error(f"爬虫执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()
