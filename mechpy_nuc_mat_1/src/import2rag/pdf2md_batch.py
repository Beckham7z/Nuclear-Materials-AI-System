import os
import fitz  # PyMuPDF
import base64
from PIL import Image
import io

def pdf_to_markdown(pdf_path, output_dir):
    """将单个PDF转换为Markdown并提取图片"""
    
    # 获取PDF文件名（不含扩展名）作为子目录名
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(output_dir, pdf_filename)
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    images_dir = os.path.join(pdf_output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    
    markdown_content = []
    image_count = 0
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # 提取文本
        text = page.get_text()
        if text.strip():
            markdown_content.append(f"## 第 {page_num + 1} 页\n\n")
            markdown_content.append(text)
            markdown_content.append("\n\n---\n\n")
        
        # 提取图片
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"image_{page_num + 1}_{img_index + 1}.{image_ext}"
                image_path = os.path.join(images_dir, image_filename)
                
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                
                markdown_content.append(f"![图片 {page_num + 1}-{img_index + 1}](images/{image_filename})\n\n")
                image_count += 1
                
            except Exception as e:
                print(f"提取第 {page_num + 1} 页图片 {img_index + 1} 时出错: {e}")
    
    pdf_document.close()
    
    # 保存markdown文件
    md_file_path = os.path.join(pdf_output_dir, f"{pdf_filename}.md")
    with open(md_file_path, "w", encoding="utf-8") as md_file:
        md_file.writelines(markdown_content)
    
    print(f"✅ {pdf_filename} 转换完成!")
    print(f"📄 Markdown文件: {md_file_path}")
    print(f"🖼️  提取了 {image_count} 张图片")
    print(f"📊 处理了 {len(pdf_document)} 页\n")
    
    return md_file_path

def batch_convert_pdfs(input_dir, output_root_dir):
    """批量转换目录下的所有PDF文件"""
    # 创建总输出目录
    os.makedirs(output_root_dir, exist_ok=True)
    
    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"正在处理: {filename}")
            try:
                pdf_to_markdown(pdf_path, output_root_dir)
            except Exception as e:
                print(f"💥 转换 {filename} 失败: {e}\n")

def main():
    """主函数"""
    # PDF文件所在目录（根据你的路径修改）
    input_dir = "/home/zyx/A_project/nuc_mat_DB/10.1016"
    # 转换结果输出根目录
    output_root_dir = "/home/zyx/A_project/mechpy/data/processed/markdown"
    
    print("=" * 60)
    print("🔄 PDF 批量转 Markdown 转换器 (PyMuPDF版本)")
    print("=" * 60)
    print(f"📂 输入目录: {input_dir}")
    print(f"📁 输出根目录: {output_root_dir}")
    print("-" * 60)
    
    try:
        batch_convert_pdfs(input_dir, output_root_dir)
        print("-" * 60)
        print(f"🎉 所有PDF文件处理完成!")
    except Exception as e:
        print(f"💥 批量处理失败: {e}")
    finally:
        print("🏁 程序执行完毕")

if __name__ == "__main__":
    main()