#Imported train.json data from Repository
import json
data=[json.loads(line) for line in open('train.json','r',encoding="cp866")]


# In[78]:


#Creating relation dict
relation_dict={}
for json in data:
    passage=json["passages"]
    #print(type(p))
    for psgs in passage:
        templ=[]
        for p in psgs["exhaustivelyAnnotatedProperties"]:
            templ.append(p["propertyName"])
            if p["propertyId"] not in relation_dict:
                relation_dict[p["propertyId"]]=p["propertyName"]
                print(p["propertyId"],p["propertyName"])


# In[119]:


relation_dict


# In[124]:


allowed_properties=['DATE_OF_BIRTH','PLACE_OF_RESIDENCE','PLACE_OF_BIRTH','NATIONALITY','EMPLOYEE_OR_MEMBER_OF','EDUCATED_AT']
allowed_property_id=[k for k,v in relation_dict.items() if  v in allowed_properties ]


# In[125]:


allowed_property_id #  all these sentences are required to make KG


# In[120]:


relation_dict.keys() #Checking the keys


# In[128]:


#Extracting Subset
subset_sentences=[]
for json in data:
    passage=json["passages"]
    #print(type(p))
    for psgs in passage:
        for f in psgs["facts"]:
            if f['propertyId'] in allowed_property_id:
                subset_sentences.append(f)


# In[129]:


len(subset_sentences)


# In[130]:


subset_sentences


# In[136]:


import pandas as pd
relationship=pd.DataFrame()
subject_list=[s["subjectText"] for s in subset_sentences]
object_list=[s["objectText"] for s in subset_sentences]
relation=[relation_dict[s["propertyId"]]for s in subset_sentences]


# In[137]:


relationship["subject"]=subject_list
relationship["object"]=object_list
relationship["relation"]=relation


# In[140]:


relationship.shape


# In[139]:


relationship.head(20)


# In[152]:


#Removing Pronouns
pronoun_list = ['I', 'me','my','his', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'they', 'them', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves', 'who', 'whom', 'whose', 'which', 'what', 'that']


# In[145]:


counts = relationship["subject"].value_counts().reset_index(name='counts')
counts.head(20)


# In[153]:


df = relationship[~relationship['subject'].str.lower().isin([pronoun.lower() for pronoun in pronoun_list])]
df.shape


# In[154]:


#Storing the final relation in a Dataframe which could then be loaded to a Knowledge Graph.
df


# In[155]:


df.to_csv("relationship.csv")

