#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#                     _ooOoo_
#                   o8888888o
#                    88" . "88
#                 ( | -  _  - | )
#                     O\ = /O
#                 ____/`---'\____
#                  .' \\| |// `.
#                 / \\|||:|||// \
#               / _|||||-:- |||||- \
#                | | \\\ - /// | |
#              | \_| ''\---/'' | _/ |
#               \ .-\__ `-` ___/-. /
#            ___`. .' /--.--\ `. . __
#         ."" '< `.___\_<|>_/___.' >'"".
#       | | : `- \`.;`\  _ /`;.`/ - ` : | |
#          \ \ `-. \_ __\ /__ _/ .-` / /
#      ==`-.____`-.___\_____/___.-`____.-'==
#                     `=---='
'''
@Project ：pythonalgorithms 
@File ：BinaryTree.py
@Author ：不胜人生一场醉
@Date ：2021/7/15 23:35 
'''

from graphviz import Digraph
import uuid
from random import sample

class Tree:
    """abstract base class representing a tree structure"""

    # ----------------------nested Position class----------------
    class Position:
        """an abstraction reprenting the location of a single element"""

        def element(self):
            """return the element stored at this Position"""
            raise NotImplementedError('must be implemented by subclass')

        def __eq__(self, other):
            """return true if other Position represents the same location"""
            raise NotImplementedError('must be implemented by subclass')

        def __ne__(self, other):
            return not (self == other)

    # --------------abstract methods that concrete subclass must support------
    def root(self):
        """return position representing the tree's root (None if empty)"""
        raise NotImplementedError('must be implemented by subclass')

    def parent(self, p):
        """Return Position representing p s parent (or None if p is root)"""
        raise NotImplementedError('must be implemented by subclass')

    def num_children(self, p):
        """Return the number of children that Position p has"""
        raise NotImplementedError('must be implemented by subclass')

    def children(self, p):
        """Generate an iteration of Positions representing p s children"""
        raise NotImplementedError('must be implemented by subclass')

    def __len__(self):
        """Return the total number of elements in the tree"""
        raise NotImplementedError('must be implemented by subclass')

    # ---------- concrete methods implemented in this class ----------
    def is_root(self, p):
        return self.root() == p

    def is_leaf(self, p):
        return self.num_children(p) == 0

    def is_empty(self):
        return len(self) == 0

    def depth(self, p):
        """Return the number of levels separating Position p from the root."""
        if self.is_root(p):
            return 0
        else:
            return self.depth(self.parent(p)) + 1

    def _height1(self):
        """return the height of the tree"""
        return max(self.depth(p) for p in self.positions() if self.is_leaf(p))

    def _height2(self, p):
        """return the height of the subtree rooted at position p"""
        if self.is_leaf(p):
            return 0
        else:
            return max(self._height2(c) for c in self.children(p)) + 1

    def height(self, p=None):
        """
        return the height of the subtree rooted at postion p.
        if p is None, return the height of the entire tree
        """
        if p is None:
            p = self.root()
        return self._height2(p)



# 二叉树类
class BinaryTree(object):
    # 初始化
    def __init__(self, root=None):
        self.data = root  # 数据
        self.leftchild = None  # 左子树
        self.rightchild = None  # 右子树
        self.dot = Digraph(comment='Binary Tree')

    # 插入左节点
    def insertleft(self, data):
        newnode = BinaryTree(data)
        if self.leftchild == None:
            self.leftchild = newnode
        else:
            newnode.leftchild = self.leftchild
            self.leftchild = newnode
        return newnode

    # 插入右节点
    def insertright(self, data):
        newnode = BinaryTree(data)
        if self.rightchild == None:
            self.rightchild = newnode
        else:
            newnode.rightchild = self.rightchild
            self.rightchild = newnode
        return newnode

    # 二叉树的高度
    def height(self):
        # 空的树高度为0, 只有root节点的树高度为1
        if self.data is None:
            return 0
        elif self.leftchild == None and self.rightchild == None:
            return 1
        elif self.leftchild == None and self.rightchild != None:
            return 1 + self.rightchild.height()
        elif self.leftchild != None and self.rightchild == None:
            return 1 + self.leftchild.height()
        else:
            return 1 + max(self.leftchild.height(), self.rightchild.height())

    # 访问右子节点
    def getrightchild(self):
        return self.rightchild

    # 访问左子节点
    def getleftchild(self):
        return self.leftchild

    # 设置根节点的值
    def setrootvalue(self, data):
        self.data = data

    # 获取根节点的值
    def getrootvalue(self):
        return self.data

    # 前序遍历
    def preorder(self, list_data: list):

        if self.data != None:
            #print(self.data, end=' ')
            list_data.append(self.data)
        if self.leftchild != None:
            self.leftchild.preorder(list_data)
        if self.rightchild != None:
            self.rightchild.preorder(list_data)
        return(list_data)

    # 中序遍历
    def inorder(self):

        if self.leftchild != None:
            self.leftchild.inorder()
        if self.data != None:
            print(self.data, end=' ')
        if self.rightchild != None:
            self.rightchild.inorder()

    # 后序遍历
    def postorder(self):

        if self.leftchild != None:
            self.leftchild.postorder()
        if self.rightchild != None:
            self.rightchild.postorder()
        if self.data != None:
            print(self.data, end=' ')

    # 利用Graphviz实现二叉树的可视化
    def printtree(self, save_path='./Binary_Tree.gv', nodee='', label=False):

        # colors for labels of nodes
        colors = ['skyblue', 'tomato', 'orange', 'purple', 'green', 'yellow', 'pink', 'red']
        op_type_color = {}
        op_type_color['Conv'] = 'purple'
        op_type_color['Relu'] = 'orange'
        op_type_color['MaxPool'] = 'tomato'
        op_type_color['GlobalAveragePool'] = 'skyblue' 
        op_type_color['AveragePool'] = 'skyblue'
        op_type_color['Flatten'] = 'pink'
        op_type_color['Gemm'] = 'tomato'
        op_type_color['Add'] = 'purple'
        op_type_color['Concat'] = 'yellow'
        op_type_color['Pad'] = 'yellow'

        # 绘制以某个节点为根节点的二叉树
        def print_node(node, node_tag):
            # 节点颜色
            color = sample(colors, 1)[0]
            if node.leftchild is not None:
                left_tag = str(uuid.uuid1())  # 左节点的数据
                this_optype = nodee[node.leftchild.data].op_type
                color = op_type_color[this_optype]
                self.dot.node(left_tag, '['+str(node.leftchild.data)+']: '+str(this_optype), style='filled', fontsize = '24', fontcolor = color,color=color)  # 左节点
                label_string = 'L' if label else ''  # 是否在连接线上写上标签，表明为左子树
                self.dot.edge(node_tag, left_tag, label=label_string)  # 左节点与其父节点的连线
                print_node(node.leftchild, left_tag)

            if node.rightchild is not None:
                right_tag = str(uuid.uuid1())
                this_optype = nodee[node.rightchild.data].op_type
                color = op_type_color[this_optype]
                self.dot.node(right_tag, '['+str(node.rightchild.data)+']: '+str(this_optype), style='filled', fontsize = '24', fontcolor = color,color=color)  # 右节点
                label_string = 'R' if label else ''  # 是否在连接线上写上标签，表明为右子树
                self.dot.edge(node_tag, right_tag, label=label_string)
                print_node(node.rightchild, right_tag)

        # 如果树非空
        if self.data is not None:
            root_tag = str(uuid.uuid1())  # 根节点标签
            this_optype = nodee[self.data].op_type
            color = op_type_color[this_optype]
            self.dot.node(root_tag, '['+str(self.data)+']: '+str(this_optype), style='filled', fontsize = '24', fontcolor = color, color=color)  # 创建根节点
            print_node(self, root_tag)

        self.dot.render(save_path)  # 保存文件为指定文件


if __name__ == "__main__":
    r = BinaryTree('a')
    rb = r.insertleft('b')
    print(r.getleftchild().getrootvalue())
    rc = r.insertright('c')
    rbd = rb.insertleft('d')
    rbe = rb.insertright('e')
    rcf = rc.insertleft('f')
    print('先序遍历为:')
    dd = r.preorder()

    print('中序遍历为:')
    r.inorder()
    print()

    print('后序遍历为:')
    r.postorder()
    print()

    # rcfg = rcf.insertleft('g')
    # print('先序遍历为:')
    # r.preorder()
    # print()
    rcfg = rc.insertleft('g')
    print('先序遍历为:')
    r.preorder()
    print()

    r.setrootvalue('aaa')
    print('先序遍历为:')
    r.preorder()
    print()
    print('根节点的值为')
    print(r.getrootvalue())
    print('树的高度是:')
    print(r.height())

    r.printtree()