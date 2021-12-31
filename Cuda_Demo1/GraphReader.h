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
		static int Init_pages(Node* &nodes, double* &pages_probs, int* &edges_list, int& dangling_nodes_count);//��ʼ��ÿ��page�����ֵ

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

		//�������ݸ�ʽ��һ����page��,���������Ϊ��node��
		int to, from;
		iss >> nodes_count;
		printf("�Լ������pages_count : %d\n", nodes_count);

		edges_list = vector<vector<int> >(nodes_count);
		links = vector<int>(nodes_count), 0;

		while (getline(file, line)) {

			stringstream inp(line);
			inp >> from >> to;
			//cout << from << "  " << to << endl;
			//��¼to�����
			edges_list[to].push_back(from);
			edges_list[from].push_back(to);
			//��¼�ж�������
			edges_count+=2;

		}
		
	
		file.close();
	}


	int GraphReader::Init_pages(Node* &nodes, double* &pages_probs, int* &Adj_nodes, int& dangling_nodes_count) {

		Adj_nodes = new int[edges_count];
		pages_probs = new double[nodes_count];

		//��page�ṹ��ȥ����page_count��page
		nodes = new Node[nodes_count];

		dangling_nodes_count = 0;

		// Initialize the pages_probs (I) vector with 1/n values
		//������page����ֵΪ1/n
		for (int i = 0; i < nodes_count; ++i) {
			pages_probs[i] = 1.0 / nodes_count;
		}

		int next_idx = 0;
		for (int i = 0; i < nodes_count; ++i) {
			nodes[i].links_count = int(edges_list[i].size());


			if (edges_list[i].size() > 0) { //start_idx �� end_idx ��¼��in_node�е��±꣬��ֱ������Ѱַ
				nodes[i].Adjstart_index = next_idx;
				nodes[i].Adjend_index = int(edges_list[i].size()) + next_idx;
			}

			for (int j = 0; j < edges_list[i].size(); j++) {
				int  from = edges_list[i][j];
				Adj_nodes[next_idx++] = from;   // ��¼i����ڽӵ�����Щ��
			}

			//��ǳ���Ϊ0�ĵ�
			if (links[i] == 0) {
				nodes[i].dangling_node = true;
				dangling_nodes_count++;
			}
		}

		return nodes_count;
	}

}

#endif /* GRAPHREADER_H_ */
