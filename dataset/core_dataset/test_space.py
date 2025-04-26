import json

def detect_zero_width_spaces(file_path):
    """
    检测JSON文件中的记录是否包含零宽空格，并返回相应结果
    
    参数:
    file_path (str): 包含记录数组的JSON文件路径
    
    返回:
    list: 包含每条记录分析结果的列表，1=chosen含有零宽空格而rejected没有，
         0=rejected含有零宽空格而chosen没有，其他情况返回描述字符串
    """
    results = []
    
    try:
        # 读取整个文件内容为一个JSON对象
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            records = json.loads(file_content)
            
            # 确保records是可迭代的
            if not isinstance(records, list):
                print("警告: 文件内容不是JSON数组，尝试使用单个记录进行处理")
                records = [records]
            
            for idx, record in enumerate(records):
                if 'chosen' not in record or 'rejected' not in record:
                    print(f"记录 {idx}: 缺少'chosen'或'rejected'字段，跳过")
                    continue
                
                # 提取需要检查的文本，处理不同可能的数据结构
                if isinstance(record['chosen'], dict):
                    chosen_text = record['chosen'].get('value', '')
                elif isinstance(record['chosen'], str):
                    chosen_text = record['chosen']
                else:
                    chosen_text = str(record['chosen'])
                
                if isinstance(record['rejected'], dict):
                    rejected_text = record['rejected'].get('value', '')
                elif isinstance(record['rejected'], str):
                    rejected_text = record['rejected']
                else:
                    rejected_text = str(record['rejected'])
                
                # 检查零宽空格
                chosen_has_zws = '\u200B' in chosen_text
                rejected_has_zws = '\u200B' in rejected_text
                
                # 确定结果
                if chosen_has_zws and not rejected_has_zws:
                    result = 1
                elif not chosen_has_zws and rejected_has_zws:
                    result = 0
                else:
                    # 如果两者都有或都没有
                    if chosen_has_zws and rejected_has_zws:
                        result = "两者都含有零宽空格"
                    else:
                        result = "两者都不含有零宽空格"
                
                results.append(result)
    
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print("请确保文件包含有效的JSON数据")
    except FileNotFoundError:
        print(f"找不到文件: {file_path}")
    except Exception as e:
        print(f"处理文件时出错: {e}")
    
    return results

def main():
    file_path = '/bigtemp/trv3px/malla-backdoor/dataset/dpo_dataset/dpo_dataset.json'
    results = detect_zero_width_spaces(file_path)
    
    if results:
        print("检测结果:")
        for i, result in enumerate(results):
            print(f"记录 {i+1}: {result}")

if __name__ == "__main__":
    main()