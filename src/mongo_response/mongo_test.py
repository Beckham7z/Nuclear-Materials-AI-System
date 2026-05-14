from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
coll = client['mech']['md_documents']
print('总条数:', coll.count_documents({}))
print('第一条 _id:', coll.find_one({})['_id'])