#ifndef __COREENGINE__HPP__
#define __COREENGINE__HPP__

#include<string>
#include<vector>
#include<map>
#include "invertedindex.hpp"
#include "parser.hpp"

class CoreEngine {
public:
    CoreEngine();
    ~CoreEngine();
    std::vector<std::string> populate(std::string const& dirname);
    void search(std::string& query, std::vector<std::map<int, int>>& freqs);
    int num_files();
    int i = 0;
	std::map<int, RetrievalData*> mDocs;
private:
    void process_file(std::string& filename);
    InvertedIndex mMap;
    std::vector<std::string> mFiles;
};
#endif

