# -*- coding: utf-8 -*-
import pandas as pd


def get_path(start, child_father_dict):
    """
    从叶子节点出发，不断向根节点遍历，得到所有叶子节点到根节点的路径
    :param start:
            eg: 'c'
    :param child_father_dict:
            eg: {'c': {'d', 'b'}, 'd': {'a'}, 'b': {'a'}}
    :return: all_path:
            eg: [[['c', 'b', 'a'], ['c', 'd', 'a']]]
    """

    def _dfs(node, path, res):
        path.append(node)
        if node not in child_father_dict:
            res.append(path.copy())
            path.pop()
            return
        for ni in child_father_dict[node]:
            _dfs(ni, path, res)
        path.pop()

    all_path = []
    if start in child_father_dict:
        _dfs(start, [], all_path)
    return all_path


if __name__ == '__main__':
    df = pd.DataFrame({'father': ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                       'child': ['b', 'c', 'd', 'e', 'f', 'f', 'g', 'g', 'h']})
    father_set = set(df['father'])
    child_set = set(df['child'])
    # 找到叶子节点的集合
    leaf_node_set = child_set - father_set
    # 生成子项父项字典
    chi_fat_dict = df.groupby('child').agg({'father': set}).to_dict()['father']
    print(chi_fat_dict)
    # 以叶子节点集合中的每一个结点为起始点，不断向根节点遍历，得到所有叶子节点到根节点的路径
    all_paths = [get_path(son, chi_fat_dict) for son in leaf_node_set]
    print(all_paths)

    list_a = [0]
    print(type(list_a))

    #遍历节点 1、找出所有stop节点；2、找出所有除了父节点的节点的父节点，生成字典