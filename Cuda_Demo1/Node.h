#ifndef NODE_H_
#define NODE_H_
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdio.h>

using namespace std;

/*
page这个结构，将所有点都写成这样的结构
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
