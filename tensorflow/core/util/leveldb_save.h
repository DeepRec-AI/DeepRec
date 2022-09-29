//
// Created by Administrator on 2022/7/31.
//

#ifndef CPP_TUTORIAL_LEVELDB_SAVE_H
#define CPP_TUTORIAL_LEVELDB_SAVE_H

#include "leveldb/db.h"
#include "leveldb/comparator.h"
#include "leveldb/write_batch.h"

#include <sstream>

using leveldb::DB;
using leveldb::Options;
using leveldb::ReadOptions;
using leveldb::WriteBatch;
using leveldb::WriteOptions;

namespace tensorflow{
    template <class K,class V>
    class LevelDBSave{
public:
        LevelDbSave(std::string DBPath){
            DBPath = "testdb";
            options.create_if_missing = true;
            leveldb::Status status = leveldb::DB::Open(options, DBPath, &db);
            if(!status.ok()){
                cout << "open db failed" << endl;
            }
        }
        ~LevelDbSave(){
            delete db;
        }
        void insert(K key, V value){
            string key_str = key.toString();
            string value_str = value.toString();
            db->Put(WriteOptions(), key_str, value_str);
        }
        void remove(K key){
            string key_str = key.toString();
            db->Delete(WriteOptions(), key_str);
        }
        void update(K key, V value){
            string key_str = key.toString();
            string value_str = value.toString();
            db->Put(WriteOptions(), key_str, value_str);
        }
        void query(K key){
            string keyStr = keyToString(key);
            string valueStr;
            db->Get(ReadOptions(), keyStr, &valueStr);
            cout << valueStr << endl;
        }
        void queryAll(){
            leveldb::Iterator* it = db->NewIterator(ReadOptions());
            for(it->SeekToFirst(); it->Valid(); it->Next()){
                cout << it->key() << ":" << it->value() << endl;
            }
            delete it;
        }
        void save(K key, V value){
            string keyStr = keyToString(key);
            string valueStr = valueToString(value);
            writeBatch.Put(keyStr, valueStr);
        }
        void save(K key, V value, string tableName){
            string keyStr = keyToString(key);
            string valueStr = valueToString(value);
            writeBatch.Put(tableName + keyStr, valueStr);
        }
        void save(K key, V value, string tableName, string keyPrefix){
            string keyStr = keyToString(key);
            string valueStr = valueToString(value);
            writeBatch.Put(tableName + keyPrefix + keyStr, valueStr);
        }
        void save(K key, V value, string tableName, string keyPrefix, string keySuffix){
            string keyStr = keyToString(key);
            string valueStr = valueToString(value);
            writeBatch.Put(tableName + keyPrefix + keyStr + keySuffix, valueStr);
        }
        void save(K key, V value, string tableName, string keyPrefix, string keySuffix, string keySuffix2){
            string keyStr = keyToString(key);
            string valueStr = valueToString(value);
            writeBatch.Put(tableName + keyPrefix + keyStr + keySuffix + keySuffix2, valueStr);
        }
        void save(K key, V value, string tableName, string keyPrefix, string keySuffix, string keySuffix2, string keySuffix3){
            string keyStr = keyToString(key);
            string valueStr = valueToString(value);
        }

        private:
            string keyToString(K key){
                stringstream ss;
                ss << key;
                return ss.str();
            }
            string valueToString(V value){
                stringstream ss;
                ss << value;
                return ss.str();
            }
            Options options;
            DB* db;
            WriteBatch writeBatch;
            std::string DBPath;


    };

}
#endif //CPP_TUTORIAL_LEVELDB_SAVE_H
