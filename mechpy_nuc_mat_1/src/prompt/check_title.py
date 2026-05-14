# 使用语言模型检查DOI与文章标题是否对应的提示词
#  
GET_Title_PROMPT = """You are an expert in mechanical metamaterials. I will upload the first 4000 characters of an article. 
                        Please output the title of the article and determine whether the article is related to mechanical metamaterials. 
                        Return the information in a JSON-formatted file.
                        example：
                        1. {
                                "Title":"Machine learning approaches for predicting protein-protein interactions",
                                "mechanical_metamaterials":false
                                }
                        2. {
                                "Title":"Lattice structure materials for cushioning",
                                "mechanical_metamaterials":true
                                }    
                        """


CHECK_Title_PROMPT = """I will provide two article titles, and you need to check if these titles have the same meaning. If there are differences such as the titles being in different languages, superscript/subscript errors, formatting inconsistencies, or the absence of phrases like "a review" that do not affect the main meaning, the titles should still be considered consistent.
                        - The output should be in JSON format and include the following:
                        - Compare the two titles. If they are different, return the original pdf_Title.
                        - If they are the same, return the title in a standardized format. Use LaTeX format for superscripts, subscripts, or special characters.
                        - Only return the English title.
                        - The JSON output must include both Title and consistency.
                        For example:
                        input:
                        config_title : controllable-stiffness components based on magnetorheological elastomers
                        pdf_title : an adaptive electrodynamic metamaterial for the absorption of structural vibration
                        output:
                        {"Title":an adaptive electrodynamic metamaterial for the absorption of structural vibration,"consistency":false}
                        input:
                        config_title : a novel strategy for constructing 3d dislocated chiral metamaterial with negative poisson's ratio
                        pdf_title : a novel strategy for constructing three-dimensional dislocated chiral metamaterial with negative poisson’s ratio
                        output:
                        {"Title":"a novel strategy for constructing three-dimensional dislocated chiral metamaterial with negative poisson’s ratio","consistency":false}
                        """