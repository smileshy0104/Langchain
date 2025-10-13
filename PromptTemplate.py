from langchain.prompts import PromptTemplate
#定义多变量模板
template = PromptTemplate(
  template="请评价{product}的优缺点，包括{aspect1}和{aspect2}。",
  input_variables=["product", "aspect1", "aspect2"]
)

#使用模板生成提示词
prompt_1 = template.format(product="智能手机", aspect1="电池续航", aspect2="拍照质量")
prompt_2 = template.format(product="笔记本电脑", aspect1="处理速度", aspect2="便携性")
print("提示词1:",prompt_1)
print("提示词2:",prompt_2)

prompt_template = PromptTemplate.from_template(
"请给我一个关于{topic}的{type}解释。"
)
#传入模板中的变量名
prompt = prompt_template.format(type="详细", topic="量子力学")
print(prompt)


#1.导入相关的包
from langchain_core.prompts import PromptTemplate
# 2.定义提示词模版对象
text = """
Tell me a joke
"""
prompt_template = PromptTemplate.from_template(text)
# 3.默认使用f-string进行格式化（返回格式好的字符串）
prompt = prompt_template.format()
print(prompt)