import pyTigerGraph as tg
import pandas as pd
import warnings

conn=tg.pyTigerGraph(host='xxxxxx')
conn.gsql('''
create graph liuqf_test()
''')

conn.graphname='liuqf_test'
conn.gsql('''
CREATE SCHEMA_CHANGE JOB alter_vertex_test_liuqf FOR GRAPH liuqf_test{
    ADD VERTEX person ( PRIMARY_ID name STRING, gender STRING, age INT, state STRING );
    ADD UNDIRECTED EDGE friendship ( FROM person, TO person, TO person, date STRING);
}
RUN SCHEMA_CHANGE JOB alter_vertex_test_liuqf
DROP JOB alter_vertex_test_liuqf
''')
import pandas as pd

person=pd.read_csv('person.csv')
friendship=pd.read_csv('friendship.csv')

conn.upsertVertexDataFrame(df=person,vertexType='person',v_id='name',attributes={'gender':'gender','age':'age','state':'stage'})

conn.upsertVertexDataFrame(df=friendship,sourceVertexType='person',edgeType='friendship',targetVertexType='person',
                           from_id='person1',to_id='person2',attributes={'date':'date'})

# conn.gsql('''
# drop graph liuqf_test
# ''')