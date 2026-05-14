#!/usr/bin/env python3
"""
RAG 可视化仪表板
提供Web界面来交互式查看RAG服务器内容
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import os
import time

# 添加父目录到路径以便导入模块
sys.path.append(str(Path(__file__).parent.parent))

from visualization.RAG_visualizer import RAGVisualizer
from visualization.enhanced_RAG_visualizer import EnhancedRAGVisualizer

class RAGDashboard:
    """RAG 可视化仪表板"""
    
    def __init__(self):
        # 初始化session_state
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = RAGVisualizer()
        if 'enhanced_visualizer' not in st.session_state:
            st.session_state.enhanced_visualizer = EnhancedRAGVisualizer()
        
        self.visualizer = st.session_state.visualizer
        self.enhanced_visualizer = st.session_state.enhanced_visualizer
        self.setup_page()
        
    def setup_page(self):
        """设置Streamlit页面配置"""
        st.set_page_config(
            page_title="RAG 可视化仪表板",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def run(self):
        """运行仪表板"""
        st.title("📊 RAG 服务器内容可视化仪表板")
        st.markdown("---")
        
        # 侧边栏配置
        st.sidebar.title("配置选项")
        
        # 数据路径配置
        if 'data_path' not in st.session_state:
            st.session_state.data_path = "data/RAG/myKG"
        
        data_path = st.sidebar.text_input(
            "RAG数据路径",
            value=st.session_state.data_path,
            help="RAG知识图谱数据目录路径"
        )
        
        # 更新session_state中的数据路径
        st.session_state.data_path = data_path
        self.visualizer.data_path = Path(data_path)
        self.enhanced_visualizer.data_path = Path(data_path)
        
        # 加载数据按钮
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("📥 加载数据", use_container_width=True):
                with st.spinner("正在加载RAG数据..."):
                    self.visualizer.load_data()
                    st.success("数据加载完成!")
        
        with col2:
            if st.button("🔄 增强分析", use_container_width=True):
                with st.spinner("正在执行增强分析..."):
                    self.enhanced_visualizer.load_data()
                    st.success("增强分析完成!")
        
        # 检查是否已加载数据
        if not hasattr(self.visualizer, 'documents') or self.visualizer.documents is None:
            st.info("请先点击'加载数据'按钮加载RAG数据")
            return
        
        # 主内容区域
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 概览", 
            "🔍 知识图谱", 
            "📄 文档分析", 
            "🏷️ 实体网络",
            "🚀 增强分析"
        ])
        
        with tab1:
            self.show_overview()
        
        with tab2:
            self.show_knowledge_graph()
        
        with tab3:
            self.show_document_analysis()
        
        with tab4:
            self.show_entity_network()
        
        with tab5:
            self.show_enhanced_analysis()
    
    def show_overview(self):
        """显示概览页面"""
        st.header("📈 系统概览")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # 显示关键指标
        with col1:
            if self.visualizer.documents:
                st.metric(
                    "文档数量",
                    len(self.visualizer.documents),
                    help="已处理的文档总数"
                )
        
        with col2:
            if self.visualizer.entities:
                st.metric(
                    "实体数量",
                    len(self.visualizer.entities),
                    help="提取的实体总数"
                )
        
        with col3:
            if self.visualizer.relations:
                st.metric(
                    "关系数量",
                    len(self.visualizer.relations),
                    help="实体间的关系总数"
                )
        
        with col4:
            if self.visualizer.graph_data:
                st.metric(
                    "知识图谱节点",
                    len(self.visualizer.graph_data['nodes']),
                    help="知识图谱中的节点总数"
                )
        
        with col5:
            if self.visualizer.graph_data:
                st.metric(
                    "知识图谱边",
                    len(self.visualizer.graph_data['edges']),
                    help="知识图谱中的边总数"
                )
        
        # 数据统计图表
        if (self.visualizer.documents and 
            self.visualizer.entities and 
            self.visualizer.relations):
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 文档类型分布
                doc_types = []
                for doc_id, doc in self.visualizer.documents.items():
                    content = doc.get('content', '')
                    if 'PDF' in content:
                        doc_types.append('PDF文档')
                    elif 'arxiv' in doc_id.lower():
                        doc_types.append('arXiv论文')
                    else:
                        doc_types.append('其他文档')
                
                type_counts = pd.Series(doc_types).value_counts()
                fig1 = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="文档类型分布"
                )
                st.plotly_chart(fig1, width='stretch', key="doc_type_pie")
            
            with col2:
                # 实体类型分布
                if self.visualizer.entities:
                    entity_types = []
                    for entity in self.visualizer.entities.values():
                        entity_type = entity.get('entity_type', 'unknown')
                        entity_types.append(entity_type)
                    
                    entity_type_counts = pd.Series(entity_types).value_counts()
                    fig2 = px.bar(
                        x=entity_type_counts.index,
                        y=entity_type_counts.values,
                        title="实体类型分布",
                        labels={'x': '实体类型', 'y': '数量'}
                    )
                    st.plotly_chart(fig2, width='stretch', key="entity_type_bar")
        
        # 快速操作
        st.subheader("🚀 快速操作")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 重新加载数据", use_container_width=True):
                with st.spinner("重新加载数据中..."):
                    self.visualizer.load_data()
                    st.success("数据重新加载完成!")
        
        with col2:
            if st.button("📊 生成完整报告", use_container_width=True):
                with st.spinner("生成报告中..."):
                    report = self.visualizer.generate_summary_report()
                    st.success("报告生成完成!")
        
        with col3:
            if st.button("🖼️ 导出可视化", use_container_width=True):
                with st.spinner("导出可视化中..."):
                    self.visualizer.generate_dashboard()
                    st.success("可视化导出完成!")
    
    def show_knowledge_graph(self):
        """显示知识图谱页面"""
        st.header("🔍 知识图谱可视化")
        
        if not self.visualizer.graph_data:
            st.warning("未找到知识图谱数据，请先加载数据")
            return
        
        # 配置选项
        col1, col2 = st.columns(2)
        
        with col1:
            max_nodes = st.slider(
                "最大节点数",
                min_value=10,
                max_value=200,
                value=50,
                help="限制显示的节点数量以提高性能"
            )
        
        with col2:
            layout_algorithm = st.selectbox(
                "布局算法",
                ["spring", "circular", "random", "kamada_kawai"],
                help="选择知识图谱的布局算法"
            )
        
        # 生成知识图谱
        if st.button("🔄 生成知识图谱", use_container_width=True):
            with st.spinner("生成知识图谱中..."):
                try:
                    # 实际调用visualizer的方法生成知识图谱
                    fig = self.visualizer.visualize_knowledge_graph(max_nodes=max_nodes)
                    
                    if fig:
                        st.success("知识图谱生成成功!")
                        
                        # 显示知识图谱
                        st.subheader("📊 知识图谱可视化")
                        st.plotly_chart(fig, width='stretch', key="knowledge_graph")
                        
                        # 显示知识图谱统计信息
                        nodes = self.visualizer.graph_data['nodes']
                        edges = self.visualizer.graph_data['edges']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("总节点数", len(nodes))
                        with col2:
                            st.metric("总边数", len(edges))
                        with col3:
                            st.metric("节点类型", len(set([n.get('entity_type', 'unknown') for n in nodes])))
                        
                        # 显示节点示例
                        st.subheader("节点示例")
                        node_df = pd.DataFrame(nodes[:10])  # 显示前10个节点
                        st.dataframe(node_df[['id', 'entity_type', 'description']], use_container_width=True)
                        
                        # 显示边示例
                        st.subheader("关系示例")
                        edge_df = pd.DataFrame(edges[:10])  # 显示前10条边
                        st.dataframe(edge_df[['source', 'target', 'description']], use_container_width=True)
                    else:
                        st.warning("知识图谱生成失败，请检查数据")
                        
                except Exception as e:
                    st.error(f"生成知识图谱时出错: {e}")
                    st.info("""
                    如果遇到依赖包问题，请确保已安装以下包:
                    ```bash
                    pip install networkx plotly matplotlib
                    ```
                    """)
    
    def show_document_analysis(self):
        """显示文档分析页面"""
        st.header("📄 文档分析")
        
        if not self.visualizer.documents:
            st.warning("未找到文档数据，请先加载数据")
            return
        
        # 文档统计
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_docs = len(self.visualizer.documents)
            st.metric("文档总数", total_docs)
        
        with col2:
            total_chars = sum(len(doc.get('content', '')) for doc in self.visualizer.documents.values())
            st.metric("总字符数", f"{total_chars:,}")
        
        with col3:
            avg_length = total_chars / total_docs if total_docs > 0 else 0
            st.metric("平均长度", f"{avg_length:.0f}")
        
        # 文档列表
        st.subheader("📋 文档列表")
        
        # 搜索和过滤
        search_term = st.text_input("🔍 搜索文档", placeholder="输入关键词搜索文档...")
        
        # 显示文档表格
        doc_data = []
        for doc_id, doc in self.visualizer.documents.items():
            content = doc.get('content', '')
            if search_term.lower() in content.lower() or search_term.lower() in doc_id.lower():
                doc_data.append({
                    'ID': doc_id,
                    '内容长度': len(content),
                    '文件路径': doc.get('file_path', 'unknown'),
                    '创建时间': doc.get('create_time', 'unknown'),
                    '内容预览': content[:100] + '...' if len(content) > 100 else content
                })
        
        if doc_data:
            doc_df = pd.DataFrame(doc_data)
            st.dataframe(doc_df, use_container_width=True)
        else:
            st.info("未找到匹配的文档")
        
        # 文档长度分布
        st.subheader("文档长度分布")
        doc_lengths = [len(doc.get('content', '')) for doc in self.visualizer.documents.values()]
        
        if doc_lengths:
            fig = px.histogram(
                x=doc_lengths,
                title="文档长度分布",
                labels={'x': '文档长度(字符)', 'y': '文档数量'}
            )
            st.plotly_chart(fig, width='stretch', key="doc_length_hist")
        
        # 文档详情查看
        st.subheader("📖 文档详情")
        selected_doc_id = st.selectbox(
            "选择文档查看详情",
            options=list(self.visualizer.documents.keys())[:20]  # 限制选项数量
        )
        
        if selected_doc_id:
            doc = self.visualizer.documents[selected_doc_id]
            st.text_area(
                "文档内容",
                value=doc.get('content', ''),
                height=300,
                key=f"doc_{selected_doc_id}"
            )
    
    def show_entity_network(self):
        """显示实体网络页面"""
        st.header("🏷️ 实体网络分析")
        
        if not self.visualizer.entities or not self.visualizer.relations:
            st.warning("未找到实体或关系数据，请先加载数据")
            return
        
        # 实体统计
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("实体总数", len(self.visualizer.entities))
        
        with col2:
            st.metric("关系总数", len(self.visualizer.relations))
        
        with col3:
            entity_types = set()
            for entity in self.visualizer.entities.values():
                entity_types.add(entity.get('entity_type', 'unknown'))
            st.metric("实体类型数", len(entity_types))
        
        # 实体类型分布
        st.subheader("实体类型分布")
        entity_type_counts = {}
        for entity in self.visualizer.entities.values():
            entity_type = entity.get('entity_type', 'unknown')
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        if entity_type_counts:
            fig = px.bar(
                x=list(entity_type_counts.keys()),
                y=list(entity_type_counts.values()),
                title="实体类型分布",
                labels={'x': '实体类型', 'y': '数量'}
            )
            st.plotly_chart(fig, width='stretch', key="entity_type_distribution")
        
        # 实体列表
        st.subheader("🔍 实体列表")
        
        # 按类型过滤
        entity_types = list(set([e.get('entity_type', 'unknown') for e in self.visualizer.entities.values()]))
        selected_type = st.selectbox("筛选实体类型", ["全部"] + entity_types)
        
        # 显示实体表格
        entity_data = []
        for entity_id, entity in self.visualizer.entities.items():
            entity_type = entity.get('entity_type', 'unknown')
            if selected_type == "全部" or entity_type == selected_type:
                entity_data.append({
                    'ID': entity_id,
                    '类型': entity_type,
                    '描述': entity.get('description', '')[:100] + '...' if len(entity.get('description', '')) > 100 else entity.get('description', ''),
                    '源ID': entity.get('source_id', ''),
                    '文件路径': entity.get('file_path', '')
                })
        
        if entity_data:
            entity_df = pd.DataFrame(entity_data)
            st.dataframe(entity_df, use_container_width=True)
        else:
            st.info("未找到匹配的实体")
        
        # 关系分析
        st.subheader("🔗 关系分析")
        
        # 显示关系表格
        relation_data = []
        for relation_id, relation in list(self.visualizer.relations.items())[:20]:  # 限制数量
            relation_data.append({
                'ID': relation_id,
                '源实体': relation.get('source_id', ''),
                '目标实体': relation.get('target_id', ''),
                '描述': relation.get('description', ''),
                '权重': relation.get('weight', 1.0)
            })
        
        if relation_data:
            relation_df = pd.DataFrame(relation_data)
            st.dataframe(relation_df, use_container_width=True)
    
    def show_enhanced_analysis(self):
        """显示增强分析页面"""
        st.header("🚀 增强分析")
        
        if not hasattr(self.enhanced_visualizer, 'graph_data') or not self.enhanced_visualizer.graph_data:
            st.warning("请先点击'增强分析'按钮执行深度分析")
            return
        
        # 增强分析概览
        st.subheader("📊 增强分析概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.enhanced_visualizer.entities:
                st.metric(
                    "文档数量",
                    len(self.enhanced_visualizer.entities),
                    help="分析的文档总数"
                )
        
        with col2:
            if self.enhanced_visualizer.graph_data:
                st.metric(
                    "实体数量",
                    len(self.enhanced_visualizer.graph_data['nodes']),
                    help="分类后的实体总数"
                )
        
        with col3:
            if self.enhanced_visualizer.graph_data:
                st.metric(
                    "关系数量",
                    len(self.enhanced_visualizer.graph_data['edges']),
                    help="分析的关系总数"
                )
        
        with col4:
            if self.enhanced_visualizer.graph_data:
                entity_types = set([node['entity_type'] for node in self.enhanced_visualizer.graph_data['nodes']])
                st.metric(
                    "实体类型",
                    len(entity_types),
                    help="智能分类的实体类型数"
                )
        
        # 知识图谱解释
        st.subheader("🧠 知识图谱智能解释")
        
        if st.button("🔍 生成智能解释", use_container_width=True):
            with st.spinner("正在分析知识图谱..."):
                explanation = self.enhanced_visualizer.explain_knowledge_graph()
                
                if explanation:
                    st.success("智能解释生成完成!")
                    
                    # 显示解释结果
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 实体分类统计")
                        entity_categories = explanation.get('entity_categories', {})
                        if entity_categories:
                            entity_df = pd.DataFrame({
                                '实体类型': list(entity_categories.keys()),
                                '数量': list(entity_categories.values())
                            })
                            fig1 = px.pie(
                                entity_df, 
                                values='数量', 
                                names='实体类型',
                                title="实体类型分布"
                            )
                            st.plotly_chart(fig1, width='stretch', key="enhanced_entity_pie")
                    
                    with col2:
                        st.subheader("🔗 关系模式分析")
                        relationships = explanation.get('relationships', {})
                        if relationships:
                            rel_df = pd.DataFrame({
                                '关系模式': list(relationships.keys()),
                                '频次': list(relationships.values())
                            })
                            fig2 = px.bar(
                                rel_df,
                                x='关系模式',
                                y='频次',
                                title="关系模式分布"
                            )
                            st.plotly_chart(fig2, width='stretch', key="enhanced_relationship_bar")
                    
                    # 关键洞察
                    st.subheader("💡 关键洞察")
                    insights = explanation.get('key_insights', [])
                    if insights:
                        for i, insight in enumerate(insights, 1):
                            st.info(f"{i}. {insight}")
                    else:
                        st.info("暂无关键洞察")
        
        # 增强版知识图谱可视化
        st.subheader("🎨 增强版知识图谱可视化")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_nodes = st.slider(
                "最大节点数",
                min_value=10,
                max_value=100,
                value=30,
                help="限制显示的节点数量以提高性能",
                key="enhanced_max_nodes"
            )
        
        with col2:
            if st.button("🔄 生成增强图谱", use_container_width=True):
                with st.spinner("生成增强版知识图谱中..."):
                    try:
                        fig = self.enhanced_visualizer.visualize_enhanced_knowledge_graph(max_nodes=max_nodes)
                        
                        if fig:
                            st.success("增强版知识图谱生成成功!")
                            st.plotly_chart(fig, width='stretch', key="enhanced_knowledge_graph")
                            
                            # 显示节点详情
                            st.subheader("📋 节点详情")
                            nodes = self.enhanced_visualizer.graph_data['nodes'][:10]  # 显示前10个节点
                            node_data = []
                            for node in nodes:
                                node_data.append({
                                    '实体名称': node['id'],
                                    '实体类型': node['entity_type'],
                                    '描述': node['description'],
                                    '文档ID': node.get('doc_id', '')
                                })
                            
                            if node_data:
                                node_df = pd.DataFrame(node_data)
                                st.dataframe(node_df, use_container_width=True)
                        else:
                            st.warning("增强版知识图谱生成失败")
                            
                    except Exception as e:
                        st.error(f"生成增强版知识图谱时出错: {e}")
        
        # 综合报告生成
        st.subheader("📄 综合报告")
        
        if st.button("📊 生成综合报告", use_container_width=True):
            with st.spinner("生成综合报告中..."):
                report = self.enhanced_visualizer.generate_comprehensive_report()
                
                # 显示报告
                st.subheader("📋 报告内容")
                st.markdown(report)
                
                # 提供下载
                st.download_button(
                    label="📥 下载报告",
                    data=report,
                    file_name="RAG_comprehensive_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        
        # 性能优化建议
        st.subheader("⚡ 性能优化建议")
        
        if st.button("💡 生成优化建议", use_container_width=True):
            with st.spinner("分析性能优化建议中..."):
                # 基于当前数据生成优化建议
                suggestions = []
                
                if self.enhanced_visualizer.graph_data:
                    nodes = self.enhanced_visualizer.graph_data['nodes']
                    edges = self.enhanced_visualizer.graph_data['edges']
                    
                    # 分析节点密度
                    if len(nodes) > 50:
                        suggestions.append("节点数量较多，建议使用分层布局或过滤显示")
                    
                    # 分析关系复杂度
                    if len(edges) / len(nodes) > 3:
                        suggestions.append("关系复杂度较高，建议优化实体提取策略")
                    
                    # 分析实体类型分布
                    entity_types = [node['entity_type'] for node in nodes]
                    type_counts = pd.Series(entity_types).value_counts()
                    if len(type_counts) < 3:
                        suggestions.append("实体类型较少，建议丰富实体提取规则")
                
                if suggestions:
                    for i, suggestion in enumerate(suggestions, 1):
                        st.warning(f"{i}. {suggestion}")
                else:
                    st.success("当前知识图谱结构良好，无需特殊优化")

def main():
    """主函数"""
    dashboard = RAGDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
