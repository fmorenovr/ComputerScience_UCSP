#ifndef __PARSER__
#define __PARSER__

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

struct RetrievalData{
    std::string contain;
    std::string file_location;
    std::string title;
    unsigned int db_index;
	RetrievalData(int _db_i, std::string _t, std::string c, std::string f_l):db_index(_db_i), title(_t),contain(c), file_location(f_l){};
};

class Parser {
private:
	std::string File_dir;
	bool getHead(const std::string & s);
	bool get_eo_doc(const std::string & s);
	void get_value_of(const std::string & query, const std::string & line, std::string & val);
	void get_value_of(const std::string & query, const std::string & line, unsigned int & val);
public:
	Parser(std::string f) : File_dir(f) {};
	bool getNextDocument(std::ifstream & file_open, std::vector<RetrievalData> & ans);
	void getDocuments(std::vector<RetrievalData> & ans);
};
#endif
