#ifndef NODE_H_
#define NODE_H_
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdio.h>

using namespace std;

/*
page����ṹ�������е㶼д�������Ľṹ
*/
struct Node {
	bool dangling_node;

	int links_count;
	

	int Adjstart_index;
	int Adjend_index;

	Node() {
		dangling_node = false;
		links_count = 0;
		Adjstart_index = Adjend_index = -1;
	}
};

#endif /* NODE_H_ */
