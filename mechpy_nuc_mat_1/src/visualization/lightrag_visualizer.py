#!/usr/bin/env python3
"""
RAG 服务器内容可视化工具
用于可视化RAG知识图谱、文档和实体关系
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os

class LightRAGVisualizer:
    """RAG 数据可视化器"""
    
    def __init__(self, data_path: str = "data/lightrag/myKG"):
        """
        初始化可视化器
        
        Args:
            data_path: RAG数据目录路径
        """
        self.data_path = Path(data_path)
        self.graph_data = None
        self.documents = None
        self.entities = None
        self.relations = None
        
    def load_data(self):
        """加载所有RAG数据"""
        print("正在加载RAG数据...")
        
        # 加载知识图谱
        graphml_path = self.data_path / "graph_chunk_entity_relation.graphml"
        if graphml_path.exists():
            self.graph_data = self._load_graphml(graphml_path)
            print(f"✓ 加载知识图谱: {len(self.graph_data['nodes'])} 个节点, {len(self.graph_data['edges'])} 条边")
        
        # 加载文档
        docs_path = self.data_path / "kv_store_full_docs.json"
        if docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"✓ 加载文档: {len(self.documents)} 个文档")
        
        # 加载实体
        entities_path = self.data_path / "kv_store_full_entities.json"
        if entities_path.exists():
            with open(entities_path, 'r', encoding='utf-8') as f:
                self.entities = json.load(f)
            print(f"✓ 加载实体: {len(self.entities)} 个实体")
        
        # 加载关系
        relations_path = self.data_path / "kv_store_full_relations.json"
        if relations_path.exists():
            with open(relations_path, 'r', encoding='utf-8') as f:
                self.relations = json.load(f)
            print(f"✓ 加载关系: {len(self.relations)} 个关系")
        
        print("数据加载完成!")
        
    def _load_graphml(self, file_path: Path) -> Dict[str, Any]:
        """加载GraphML文件"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 定义命名空间（默认命名空间）
            ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
            
            # 解析键定义
            keys = {}
            for key in root.findall('.//g:key', ns):
                keys[key.get('id')] = {
                    'name': key.get('attr.name'),
                    'type': key.get('attr.type'),
                    'for': key.get('for')
                }
            
            # 解析节点和边
            graph = root.find('.//g:graph', ns)
            if graph is None:
                print(f"警告: 在文件 {file_path} 中未找到graph元素")
                return {'nodes': [], 'edges': [], 'keys': keys}
            
            nodes = []
            edges = []
            
            for node in graph.findall('.//g:node', ns):
                node_data = {'id': node.get('id')}
                for data in node.findall('.//g:data', ns):
                    key_id = data.get('key')
                    if key_id in keys:
                        key_name = keys[key_id]['name']
                        node_data[key_name] = data.text
                nodes.append(node_data)
            
            for edge in graph.findall('.//g:edge', ns):
                edge_data = {
                    'source': edge.get('source'),
                    'target': edge.get('target')
                }
                for data in edge.findall('.//g:data', ns):
                    key_id = data.get('key')
                    if key_id in keys:
                        key_name = keys[key_id]['name']
                        edge_data[key_name] = data.text
                edges.append(edge_data)
            
            print(f"成功解析: {len(nodes)} 个节点, {len(edges)} 条边")
            return {'nodes': nodes, 'edges': edges, 'keys': keys}
        except ET.ParseError as e:
            print(f"错误: 解析GraphML文件失败: {e}")
            return {'nodes': [], 'edges': [], 'keys': {}}
        except Exception as e:
            print(f"错误: 加载GraphML文件时发生异常: {e}")
            return {'nodes': [], 'edges': [], 'keys': {}}
    
    def visualize_knowledge_graph(self, max_nodes: int = 50):
        """可视化知识图谱"""
        if not self.graph_data:
            print("未找到知识图谱数据")
            return
        
        print("正在生成知识图谱可视化...")
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点（限制数量）
        nodes_to_add = self.graph_data['nodes'][:max_nodes]
        for node in nodes_to_add:
            G.add_node(node['id'], **node)
        
        # 添加边（只连接已添加的节点）
        for edge in self.graph_data['edges']:
            if edge['source'] in G.nodes and edge['target'] in G.nodes:
                G.add_edge(edge['source'], edge['target'], **edge)
        
        # 使用Plotly可视化
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"节点: {node}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color='lightblue',
                line_width=2))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text='RAG 知识图谱', font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text=f"节点数: {len(G.nodes())}, 边数: {len(G.edges())}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        # 保存为HTML文件
        output_path = self.data_path.parent / "knowledge_graph_visualization.html"
        fig.write_html(str(output_path))
        print(f"✓ 知识图谱可视化已保存到: {output_path}")
        
        return fig
    
    def visualize_document_statistics(self):
        """可视化文档统计信息"""
        if not self.documents:
            print("未找到文档数据")
            return
        
        print("正在生成文档统计可视化...")
        
        # 分析文档内容
        doc_lengths = []
        doc_types = []
        
        for doc_id, doc in self.documents.items():
            content = doc.get('content', '')
            doc_lengths.append(len(content))
            
            # 简单分类文档类型
            if 'PDF' in content:
                doc_types.append('PDF文档')
            elif 'arxiv' in doc_id.lower():
                doc_types.append('arXiv论文')
            else:
                doc_types.append('其他文档')
        
        # 创建统计图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 文档长度分布
        ax1.hist(doc_lengths, bins=20, alpha=0.7, color='skyblue')
        ax1.set_title('文档长度分布')
        ax1.set_xlabel('文档长度(字符)')
        ax1.set_ylabel('文档数量')
        
        # 文档类型分布
        type_counts = Counter(doc_types)
        ax2.bar(type_counts.keys(), type_counts.values(), color='lightcoral')
        ax2.set_title('文档类型分布')
        ax2.set_xlabel('文档类型')
        ax2.set_ylabel('文档数量')
        ax2.tick_params(axis='x', rotation=45)
        
        # 文档创建时间分布
        create_times = [doc.get('create_time', 0) for doc in self.documents.values()]
        if any(create_times):
            ax3.hist(create_times, bins=20, alpha=0.7, color='lightgreen')
            ax3.set_title('文档创建时间分布')
            ax3.set_xlabel('创建时间戳')
            ax3.set_ylabel('文档数量')
        
        # 文档内容关键词统计
        all_content = ' '.join([doc.get('content', '') for doc in self.documents.values()])
        words = all_content.split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)
        
        if common_words:
            words, counts = zip(*common_words)
            ax4.bar(words, counts, color='gold')
            ax4.set_title('常见关键词')
            ax4.set_xlabel('关键词')
            ax4.set_ylabel('出现次数')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.data_path.parent / "document_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 文档统计可视化已保存到: {output_path}")
        
        plt.show()
        
        return fig
    
    def visualize_entity_network(self):
        """可视化实体网络"""
        if not self.entities or not self.relations:
            print("未找到实体或关系数据")
            return
        
        print("正在生成实体网络可视化...")
        
        # 创建实体网络
        G = nx.Graph()
        
        # 添加实体节点
        for entity_id, entity in self.entities.items():
            entity_type = entity.get('entity_type', 'unknown')
            G.add_node(entity_id, type=entity_type, **entity)
        
        # 添加关系边
        for relation_id, relation in self.relations.items():
            source = relation.get('source_id')
            target = relation.get('target_id')
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target, **relation)
        
        # 使用Plotly可视化
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 按实体类型着色
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'unknown')
            if node_type == 'data':
                node_colors.append('lightblue')
            elif node_type == 'artifact':
                node_colors.append('lightcoral')
            elif node_type == 'method':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_data = G.nodes[node]
            node_text.append(f"实体: {node}<br>类型: {node_data.get('type', 'unknown')}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color=node_colors,
                line_width=2))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text='RAG 实体网络', font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text=f"实体数: {len(G.nodes())}, 关系数: {len(G.edges())}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        # 保存为HTML文件
        output_path = self.data_path.parent / "entity_network_visualization.html"
        fig.write_html(str(output_path))
        print(f"✓ 实体网络可视化已保存到: {output_path}")
        
        return fig
    
    def generate_dashboard(self):
        """生成完整的可视化仪表板"""
        print("正在生成RAG可视化仪表板...")
        
        # 加载数据
        self.load_data()
        
        # 生成各种可视化
        visualizations = []
        
        if self.graph_data:
            kg_fig = self.visualize_knowledge_graph()
            visualizations.append(("知识图谱", kg_fig))
        
        if self.documents:
            doc_fig = self.visualize_document_statistics()
            visualizations.append(("文档统计", doc_fig))
        
        if self.entities and self.relations:
            entity_fig = self.visualize_entity_network()
            visualizations.append(("实体网络", entity_fig))
        
        # 生成汇总报告
        self.generate_summary_report()
        
        print("✓ RAG可视化仪表板生成完成!")
        return visualizations
    
    def generate_summary_report(self):
        """生成数据汇总报告"""
        print("正在生成数据汇总报告...")
        
        report = {
            "总文档数": len(self.documents) if self.documents else 0,
            "总实体数": len(self.entities) if self.entities else 0,
            "总关系数": len(self.relations) if self.relations else 0,
            "知识图谱节点数": len(self.graph_data['nodes']) if self.graph_data else 0,
            "知识图谱边数": len(self.graph_data['edges']) if self.graph_data else 0
        }
        
        # 保存报告
        report_path = self.data_path.parent / "RAG_summary_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("数据汇总报告:")
        for key, value in report.items():
            print(f"  {key}: {value}")
        
        print(f"✓ 汇总报告已保存到: {report_path}")
        
        return report

def main():
    """主函数"""
    visualizer = RAGVisualizer()
    
    print("=" * 50)
    print("RAG 服务器内容可视化工具")
    print("=" * 50)
    
    # 生成完整仪表板
    visualizer.generate_dashboard()
    
    print("\n" + "=" * 50)
    print("可视化完成! 请查看生成的HTML和图像文件")
    print("=" * 50)

if __name__ == "__main__":
    main()
