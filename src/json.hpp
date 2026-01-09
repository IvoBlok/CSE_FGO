#include <fstream>
#include <iomanip>

class JsonStreamWriter {
private:
    std::ofstream file;
    bool firstElement;
    int indentLevel;
    const int indentSize = 2;
    
    void writeIndent() {
        file << std::string(indentLevel * indentSize, ' ');
    }
    
public:
    JsonStreamWriter(const std::string& filename) 
        : file(filename), firstElement(true), indentLevel(0) {
        file << "{\n";
        indentLevel++;
    }
    
    ~JsonStreamWriter() {
        if (file.is_open()) {
            file << "\n}\n";
            file.close();
        }
    }
    
    template<typename T>
    void write(const std::string& key, const T& value) {
        if (!firstElement) {
            file << ",\n";
        }
        writeIndent();
        file << "\"" << key << "\": " << value;
        firstElement = false;
    }
    
    void startArray(const std::string& key) {
        if (!firstElement) {
            file << ",\n";
        }
        writeIndent();
        file << "\"" << key << "\": [\n";
        indentLevel++;
        firstElement = true;
    }
    
    void endArray() {
        indentLevel--;
        file << "\n";
        writeIndent();
        file << "]";
        firstElement = false;
    }
    
    void startObject(const std::string& key) {
        if (!firstElement) {
            file << ",\n";
        }
        writeIndent();
        file << "\"" << key << "\": {\n";
        indentLevel++;
        firstElement = true;
    }
    
    void endObject() {
        indentLevel--;
        file << "\n";
        writeIndent();
        file << "}";
        firstElement = false;
    }
    
    template<typename T>
    void writeArrayElement(const T& value) {
        if (!firstElement) {
            file << ",\n";
        }
        writeIndent();
        file << value;
        firstElement = false;
    }
    
    void startArrayObject() {
        if (!firstElement) {
            file << ",\n";
        }
        writeIndent();
        file << "{\n";
        indentLevel++;
        firstElement = true;
    }
    
    void endArrayObject() {
        indentLevel--;
        file << "\n";
        writeIndent();
        file << "}";
        firstElement = false;
    }
};