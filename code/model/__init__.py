from .h2gt import H2GT

# 根据配置文件中的model.name选择并创建模型
def model_factory():
    return H2GT().cuda()
