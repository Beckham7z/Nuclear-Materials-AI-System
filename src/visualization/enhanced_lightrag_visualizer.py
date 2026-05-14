#!/usr/bin/env python3
"""
增强版RAG可视化器
提供更直观的知识图谱可视化和解释功能
"""

import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

class EnhancedLightRAGVisualizer:
    """增强版RAG可视化器"""
    
    def __init__(self, data_path: str = "data/RAG/myKG"):
        self.data_path = Path(data_path)
        self.entities = None
        self.relations = None
        self.documents = None
        self.graph_data = None
        
    def load_data(self):
        """加载RAG数据"""
        try:
            # 加载实体数据
            entities_file = self.data_path / "kv_store_full_entities.json"
            if entities_file.exists():
                with open(entities_file, 'r', encoding='utf-8') as f:
                    self.entities = json.load(f)
            
            # 加载关系数据
            relations_file = self.data_path / "kv_store_full_relations.json"
            if relations_file.exists():
                with open(relations_file, 'r', encoding='utf-8') as f:
                    self.relations = json.load(f)
            
            # 加载文档数据
            docs_file = self.data_path / "kv_store_full_docs.json"
            if docs_file.exists():
                with open(docs_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            self._build_graph_data()
            return True
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False
    
    def _build_graph_data(self):
        """构建图数据结构"""
        if not self.entities or not self.relations:
            return
        
        nodes = []
        edges = []
        
        # 收集所有实体作为节点
        for doc_id, entity_data in self.entities.items():
            for entity_name in entity_data.get('entity_names', []):
                node_type = self._classify_entity(entity_name)
                nodes.append({
                    'id': entity_name,
                    'entity_type': node_type,
                    'description': self._get_entity_description(entity_name),
                    'doc_id': doc_id
                })
        
        # 收集所有关系作为边
        for doc_id, relation_data in self.relations.items():
            for source, target in relation_data.get('relation_pairs', []):
                edges.append({
                    'source': source,
                    'target': target,
                    'description': f"{source} -> {target}",
                    'doc_id': doc_id
                })
        
        self.graph_data = {
            'nodes': nodes,
            'edges': edges
        }
    
    def _classify_entity(self, entity_name: str) -> str:
        """对实体进行分类"""
        entity_lower = entity_name.lower()
        
        if 'pdf' in entity_lower and '文件' in entity_lower:
            return 'PDF文档'
        elif '字节' in entity_lower or 'bytes' in entity_lower:
            return '文件大小'
        elif '内容' in entity_lower or 'content' in entity_lower:
            return '内容状态'
        elif '工具' in entity_lower or 'tool' in entity_lower:
            return '处理工具'
        elif 'pdf files' in entity_lower:
            return 'PDF文件集合'
        else:
            return '其他'
    
    def _get_entity_description(self, entity_name: str) -> str:
        """获取实体描述"""
        entity_lower = entity_name.lower()
        
        if 'pdf' in entity_lower and '文件' in entity_lower:
            return f"PDF研究文档: {entity_name}"
        elif '字节' in entity_lower or 'bytes' in entity_lower:
            return f"文件大小: {entity_name}"
        elif '内容' in entity_lower or 'content' in entity_lower:
            return "文档内容，需要专业PDF解析工具处理"
        elif '工具' in entity_lower or 'tool' in entity_lower:
            return "用于提取和处理PDF文档内容的专业工具"
        elif 'pdf files' in entity_lower:
            return "PDF文件集合"
        else:
            return entity_name
    
    def explain_knowledge_graph(self) -> Dict[str, Any]:
        """解释知识图谱内容"""
        if not self.graph_data:
            return {}
        
        explanation = {
            "summary": "知识图谱分析报告",
            "document_types": {},
            "entity_categories": {},
            "relationships": {},
            "key_insights": []
        }
        
        # 分析文档类型
        doc_types = {}
        for doc_id, entity_data in self.entities.items():
            doc_entities = entity_data.get('entity_names', [])
            for entity in doc_entities:
                if 'pdf' in entity.lower():
                    doc_types[doc_id] = "PDF研究文档"
                    break
            else:
                doc_types[doc_id] = "其他文档"
        
        explanation["document_types"] = doc_types
        
        # 分析实体类别
        entity_categories = {}
        for node in self.graph_data['nodes']:
            category = node['entity_type']
            entity_categories[category] = entity_categories.get(category, 0) + 1
        
        explanation["entity_categories"] = entity_categories
        
        # 分析关系模式
        relationship_patterns = {}
        for edge in self.graph_data['edges']:
            source_type = self._classify_entity(edge['source'])
            target_type = self._classify_entity(edge['target'])
            pattern = f"{source_type} -> {target_type}"
            relationship_patterns[pattern] = relationship_patterns.get(pattern, 0) + 1
        
        explanation["relationships"] = relationship_patterns
        
        # 生成关键洞察
        insights = []
        
        # 洞察1: 文档处理状态
        content_entities = [node for node in self.graph_data['nodes'] 
                          if node['entity_type'] == '内容状态']
        if content_entities:
            insights.append("存在需要处理的文档内容，建议使用专业PDF解析工具")
        
        # 洞察2: 文档大小分布
        size_entities = [node for node in self.graph_data['nodes'] 
                        if node['entity_type'] == '文件大小']
        if size_entities:
            insights.append(f"检测到 {len(size_entities)} 个不同大小的PDF文档")
        
        # 洞察3: 研究主题
        pdf_entities = [node for node in self.graph_data['nodes'] 
                       if node['entity_type'] == 'PDF文档']
        if pdf_entities:
            research_topics = []
            for entity in pdf_entities:
                if 'neutron' in entity['id'].lower():
                    research_topics.append("中子物理研究")
                elif 'irradiation' in entity['id'].lower():
                    research_topics.append("材料辐照研究")
                elif 'composition' in entity['id'].lower():
                    research_topics.append("材料成分研究")
            
            if research_topics:
                insights.append(f"研究主题包括: {', '.join(set(research_topics))}")
        
        explanation["key_insights"] = insights
        
        return explanation
    
    def visualize_enhanced_knowledge_graph(self, max_nodes: int = 50) -> go.Figure:
        """生成增强版知识图谱可视化"""
        if not self.graph_data:
            return None
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        nodes = self.graph_data['nodes'][:max_nodes]
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # 添加边
        edges_added = 0
        for edge in self.graph_data['edges']:
            if edge['source'] in G and edge['target'] in G:
                G.add_edge(edge['source'], edge['target'])
                edges_added += 1
                if edges_added >= max_nodes * 2:  # 限制边数量
                    break
        
        # 使用spring布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 节点颜色映射
        node_colors = {
            'PDF文档': '#FF6B6B',
            '文件大小': '#4ECDC4',
            '内容状态': '#45B7D1',
            '处理工具': '#96CEB4',
            'PDF文件集合': '#FFEAA7',
            '其他': '#DDA0DD'
        }
        
        # 创建Plotly图
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 获取节点信息
            node_info = G.nodes[node]
            node_type = node_info.get('entity_type', '其他')
            description = node_info.get('description', node)
            
            node_text.append(f"{node}<br>类型: {node_type}<br>描述: {description}")
            node_color.append(node_colors.get(node_type, '#DDA0DD'))
            
            # 根据节点类型设置大小
            size_map = {
                'PDF文档': 20,
                '文件大小': 15,
                '内容状态': 18,
                '处理工具': 16,
                'PDF文件集合': 22,
                '其他': 12
            }
            node_size.append(size_map.get(node_type, 12))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            textposition="top center",
            text=[],
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=10)
        )
        
        # 更新悬停文本
        node_trace.text = node_text
        
        # 创建图
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text='增强版知识图谱可视化<br><sub>节点颜色表示实体类型，大小表示重要性</sub>',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="知识图谱解释:<br>• 红色: PDF文档<br>• 青色: 文件大小<br>• 蓝色: 内容状态<br>• 绿色: 处理工具<br>• 黄色: PDF集合",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               align="left"
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def generate_comprehensive_report(self) -> str:
        """生成综合报告"""
        explanation = self.explain_knowledge_graph()
        
        report = "# RAG知识图谱分析报告\n\n"
        
        report += "## 1. 总体概览\n"
        report += f"- 总文档数: {len(self.entities)}\n"
        report += f"- 总实体数: {len(self.graph_data['nodes']) if self.graph_data else 0}\n"
        report += f"- 总关系数: {len(self.graph_data['edges']) if self.graph_data else 0}\n\n"
        
        report += "## 2. 实体分类统计\n"
        for category, count in explanation.get('entity_categories', {}).items():
            report += f"- {category}: {count}个\n"
        
        report += "\n## 3. 关系模式分析\n"
        for pattern, count in explanation.get('relationships', {}).items():
            report += f"- {pattern}: {count}次\n"
        
        report += "\n## 4. 关键洞察\n"
        for insight in explanation.get('key_insights', []):
            report += f"- {insight}\n"
        
        report += "\n## 5. 建议\n"
        report += "- 使用专业PDF解析工具处理标记为'内容待处理...'的文档\n"
        report += "- 关注与中子物理、材料辐照相关的研究文档\n"
        report += "- 建立更完整的研究文档元数据体系\n"
        
        return report

def main():
    """主函数"""
    visualizer = EnhancedRAGVisualizer("data/RAG/myKG")
    
    if visualizer.load_data():
        print("✓ 数据加载成功")
        
        # 生成解释
        explanation = visualizer.explain_knowledge_graph()
        print("\n知识图谱解释:")
        print(json.dumps(explanation, indent=2, ensure_ascii=False))
        
        # 生成可视化
        fig = visualizer.visualize_enhanced_knowledge_graph()
        if fig:
            fig.write_html("data/RAG/enhanced_knowledge_graph.html")
            print("✓ 增强版知识图谱已保存")
        
        # 生成报告
        report = visualizer.generate_comprehensive_report()
        with open("data/RAG/comprehensive_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("✓ 综合报告已保存")
        
    else:
        print("❌ 数据加载失败")

if __name__ == "__main__":
    main()
