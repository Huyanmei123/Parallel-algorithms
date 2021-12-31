#include "Node.h"
#include"GPUtimer.h"
#include"GraphReader.h"
#include"Kernel.h"
#include<fstream>
using namespace GraphRead_PageRank;

//-------显示各节点的邻接点信息

void Display_nodeInfo(Node *nodes,int *edges_list,int node_count) {

	for (int i = 0; i < node_count; i++) {
		cout << "index of node: " << i << " degree of node:" << nodes[i].links_count << " node of Adj: "<< endl;
		for (int j = nodes[i].Adjstart_index; j < nodes[i].Adjend_index; j++) {
			cout <<edges_list[j]<<" ";
		}
		cout << endl;
	}

}
void Out_To_File(Node *nodes, int *edges_list, int node_count, double *res) {

	ofstream file;
	file.open("D:\\殷思源\\result\\PC\\com-amazon.ungraph-334863-pc-v2.txt");
	for (int i = 0; i < node_count; i++) {
		file << fixed<< setprecision(9) <<res[i] << endl;
	}
}
void values_of_PageRank(int node_count, double *res) {
	
	for (int i = 0; i < node_count; i++) {
		cout << fixed<<setprecision(3)<<res[i]<< endl;
		
	}
	

}
void Run_pagerank( Node *nodes, double *Rank_Value, int * edges_list) {
	GPUTimer timer;
	//Display_nodeInfo(pages, edges_list, Page_count);
	Kernel page_rank(GraphReader::nodes_count, GraphReader::edges_count);
	// Allocate matrices in the gpu memory
	page_rank.allocate_data(nodes, Rank_Value, edges_list);

	// Run PageRank algorithm
	timer.start();

	page_rank.run_kernel();

	timer.stop();

	double *res = page_rank.get_result(), check_sum = 0.0;
	//values_of_PageRank(GraphReader::nodes_count, res);
	Out_To_File(nodes, edges_list, GraphReader::nodes_count, res);
	//cout << timer.elapsed();
}


int main() {
	string File_Path = "D:\\殷思源\\data\\com-amazon.ungraph-334863.txt";
	
	GraphReader::read_graph_PageRank(File_Path);
	Node *nodes; //存储每个节点
	double *Rank_Value;//存储每个节点Rank值
	int *edges_list; //
	int dangling_nodes_count; // 无邻接点记录
	int Node_count = GraphReader::Init_pages(nodes, Rank_Value, edges_list, dangling_nodes_count);
	cout << Node_count << endl;
	cout << GraphReader::edges_count << endl;
	Run_pagerank(nodes, Rank_Value, edges_list);
	//Out_To_File(nodes, edges_list, Node_count,Rank_Value);
	
	system("pause");
	return 0;
}