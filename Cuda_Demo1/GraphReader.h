#ifndef GRAPHREADER_H_
#define GRAPHREADER_H_
#include <sstream>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <iomanip>
#include "Node.h"
#include <cmath>
#include <stdio.h>

using namespace std;

typedef double* I;

namespace GraphRead_PageRank {
	class GraphReader {
	public:

		static int nodes_count;
		static int edges_count;

		static vector<int> links;
		static vector< vector<int> > edges_list;

		static void read_graph_PageRank(string path);
		static void read_graph_BC(string path);
		static int Init_pages(Node* &nodes, double* &pages_probs, int* &edges_list, int& dangling_nodes_count);//初始化每个page里面的值

	};

	int GraphReader::nodes_count = 0;
	int GraphReader::edges_count = 0;

	vector<int> GraphReader::links = vector<int>();
	vector< vector<int> > GraphReader::edges_list = vector< vector<int> >();

	void GraphReader::read_graph_PageRank(string path) {
		ifstream file;

		cout << "path is :" << path << endl;

		file.open(path.c_str());

		string line;
		getline(file, line);
		stringstream iss(line);

		//输入数据格式第一行是page数,这里可以认为是node数
		int to, from;
		iss >> nodes_count;
		printf("自己输出的pages_count : %d\n", nodes_count);

		edges_list = vector<vector<int> >(nodes_count);
		links = vector<int>(nodes_count), 0;

		while (getline(file, line)) {

			stringstream inp(line);
			inp >> from >> to;
			//cout << from << "  " << to << endl;
			//记录to的入度
			edges_list[to].push_back(from);
			edges_list[from].push_back(to);
			//记录有多少条边
			edges_count+=2;

		}
		
	
		file.close();
	}


	int GraphReader::Init_pages(Node* &nodes, double* &pages_probs, int* &Adj_nodes, int& dangling_nodes_count) {

		Adj_nodes = new int[edges_count];
		pages_probs = new double[nodes_count];

		//用page结构体去申请page_count个page
		nodes = new Node[nodes_count];

		dangling_nodes_count = 0;

		// Initialize the pages_probs (I) vector with 1/n values
		//给所有page赋初值为1/n
		for (int i = 0; i < nodes_count; ++i) {
			pages_probs[i] = 1.0 / nodes_count;
		}

		int next_idx = 0;
		for (int i = 0; i < nodes_count; ++i) {
			nodes[i].links_count = int(edges_list[i].size());


			if (edges_list[i].size() > 0) { //start_idx 与 end_idx 记录在in_node中的下标，好直接用来寻址
				nodes[i].Adjstart_index = next_idx;
				nodes[i].Adjend_index = int(edges_list[i].size()) + next_idx;
			}

			for (int j = 0; j < edges_list[i].size(); j++) {
				int  from = edges_list[i][j];
				Adj_nodes[next_idx++] = from;   // 记录i点的邻接点是哪些点
			}

			//标记出度为0的点
			if (links[i] == 0) {
				nodes[i].dangling_node = true;
				dangling_nodes_count++;
			}
		}

		return nodes_count;
	}

}

#endif /* GRAPHREADER_H_ */
