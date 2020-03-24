#ifndef __INVERTED_INDEX__HPP__
#define __INVERTED_INDEX__HPP__

#include<map>
#include<vector>
#include<string>

typedef std::vector<std::pair<int, int>> search_result;

class InvertedIndex {
public:
	void search(std::string& query, std::vector<std::map<int, int>>& freqs);
    void insert(std::string& contain, const int key);
private:
	std::string _hash(const char* word);
	std::map<std::string, std::map<int, int>> _map;
};
#endif
