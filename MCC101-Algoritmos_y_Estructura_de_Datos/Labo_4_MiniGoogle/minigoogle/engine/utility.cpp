#include "utility.hpp"
#include<chrono>
#include<cerrno>
#include<dirent.h>

using namespace std::chrono;

double profile(std::function<void(void)> func) {
    auto start = high_resolution_clock::now();
    func();
    auto finish = high_resolution_clock::now();
	return duration_cast<duration<double>>(finish - start).count();
}

std::vector<std::string> list_immediate_files(std::string const& dirname) {
	std::vector<std::string> filenames;
    DIR *target = opendir(dirname.c_str());
    if (target != NULL) {
        for (dirent *entry = readdir(target); 
             entry != NULL ;
             entry = readdir(target)) {
            if (entry->d_name[0] == '.' ||
                (entry->d_type & DT_DIR)) continue;
            filenames.push_back(dirname + "/" + std::string(entry->d_name));
        }
        closedir(target);
    }
    return filenames;
}

bool validSpecialChar(char& c) {

	switch (c)
	{
		case '�': 
		case '�':
		{
			c = 'a';
			return true;
		}
		case '�': 
		case '�':
		{
			c = 'e';
			return true;
		}
		case '�':
		case '�':
		{
			c = 'i';
			return true;
		}
		case '�':
		case '�':
		{
			c = 'o';
			return true;
		}
		case '�':
		case '�':
		{
			c = 'u';
			return true;
		}
		case '�':
		case '�': 
		{
			c = 'n';
			return true;
		}
	}

	return false;
}

std::string iso_8859_1_to_utf8(std::string &str)
{
	std::string strOut;
	for (std::string::iterator it = str.begin(); it != str.end(); ++it)
	{
		uint8_t ch = *it;
		if (ch < 0x80) {
			strOut.push_back(ch);
		}
		else {
			strOut.push_back(0xc0 | ch >> 6);
			strOut.push_back(0x80 | (ch & 0x3f));
		}
	}
	return strOut;
}
