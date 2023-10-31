import torch


def predict(model, config, input_ids, attention_mask):
    model.load_state_dict(torch.load(config.save_path))

    # result_comments = pretreatment(test_comment_list)  # 预处理去掉标点符号
    # # 转换为字id
    # tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    # result_comments_id = tokenizer(result_comments,
    #                                padding=True,
    #                                truncation=True,
    #                                max_length=120,
    #                                return_tensors='pt')
    # tokenizer_id = result_comments_id['input_ids']
    # attention_mask = result_comments_id['attention_mask']
    # # print(tokenizer_id.shape)
    # inputs = tokenizer_id

    # batch_size = input_ids.size(0)
    # batch_size = 16
    # initialize hidden state
    # h = model.init_hidden(batch_size, USE_CUDA)

    if config.use_cuda:
        model.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    model.eval()
    with torch.no_grad():
        # get the output from the model
        output = model(input_ids, attention_mask)
        # 输入： (batch_size , hidden_size)
        # 输出： (batch_size , 2) 即 对应于2个分类的概率
        output = torch.nn.Softmax(dim=1)(output)
        # print(output.shape)    torch.Size([1, 2])
        # torch.max : 返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引。
        # torch.max(output, 1)[1]：获取 每行最大值的索引，即 预测的类别 0/1
        pred = torch.max(output, 1)[1]
        # print(pred.shape)    torch.Size([1])

        # printing output value, before rounding
        # torch.max(output, 1)[0] ： 获取 每行最大值，即 预测类别的概率
        print('预测概率为: {:.6f}'.format(torch.max(output, 1)[0].item()))
        if pred.item() == 1:
            print("预测结果为:正向")
        else:
            print("预测结果为:负向")
