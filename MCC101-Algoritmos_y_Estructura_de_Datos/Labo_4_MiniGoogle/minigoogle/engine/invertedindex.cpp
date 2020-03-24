#include<cstring>
#include "utility.hpp"
#include "invertedindex.hpp"

void InvertedIndex::search(std::string& query, std::vector<std::map<int, int>>& freqs) {

	int word_pos = 0;
	char word[147];
	memset(word, 0, 147 * sizeof(char));

	for (unsigned i = 0; i < query.length(); ++i) {
		if (isalpha(query.at(i)) || validSpecialChar(query.at(i))) {
			word[word_pos++] = tolower(query.at(i));
		}
		else {
			if (word_pos > 0) {
				freqs.push_back(_map[_hash(word)]);
				word_pos = 0;
				memset(word, 0, 147 * sizeof(char));
			}
		}
	}

	if (word_pos > 0)
		freqs.push_back(_map[_hash(word)]);
}

void InvertedIndex::insert(std::string& contain, const int key) {

	int word_pos = 0;
	char word[147];
	memset(word, 0, 147 * sizeof(char));

	for (unsigned i = 0; i < contain.length(); ++i) {
		if (isalpha(contain.at(i)) || validSpecialChar(contain.at(i))) {
			word[word_pos++] = tolower(contain.at(i));
		}
		else {
			if (word_pos > 0) {
				_map[_hash(word)][key]++;
				word_pos = 0;
				memset(word, 0, 147 * sizeof(char));
			}
		}
	}

	if (word_pos > 0)
		_map[_hash(word)][key]++;
}

std::string InvertedIndex::_hash(const char* word) {
	return std::string(word);
}
