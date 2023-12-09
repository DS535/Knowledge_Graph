import streamlit as st
from PIL import Image
from neo4j import GraphDatabase
import spacy

# Connect to the Neo4j database
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Login@444"))

# Loading the English model
nlp = spacy.load("en_core_web_sm")

def extract_entity(question, entity_type):
    doc = nlp(question)
    for ent in doc.ents:
        if ent.label_ in entity_type:
            return ent.text
    return None

# Define the Streamlit app
def app():
    # Set page configuration
    st.set_page_config(
        page_title="Neo4j Query App",
        page_icon="🔍",
        layout="wide"
    )

    # Add a title and description to the app
    st.title("Chatbot using Knowledge Graph")
    st.markdown("Ask a question to the bot and it will query the Neo4j database for an answer.")

    # Add a text input for the user to enter a question
    question = st.text_input("Enter a question:")
    subject_entity = extract_entity(question, ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"])
    
    # If the user enters a question, run the Neo4j query
    if question:

        if "nationality" in question:
            with driver.session() as session:
                result = session.run(f"MATCH (s:subject)-[NATIONALITY]->(o) WHERE s.Name = '{subject_entity}' RETURN o.Name")
                for record in result:
                     if record:
                        st.success(f"The nationality of {subject_entity} is {record['o.Name']}.")
                     else:
                         st.warning("No matching subjects found.")

        elif "educated" in question:
            with driver.session() as session:
                result = session.run(f"MATCH (s:subject)-[EDUCATED_AT]->(o) WHERE s.Name = '{subject_entity}' RETURN o.Name")
                for record in result:
                     if record:
                        st.success(f"The {subject_entity} is educated at {record['o.Name']}.")
                     else:
                         st.warning("No matching subjects found.")

        elif "date of birth" in question:
            with driver.session() as session:
                result = session.run(f"MATCH (s:subject)-[DATE_OF_BIRTH]->(o) WHERE s.Name = '{subject_entity}' RETURN o.Name")
                for record in result:
                     if record:
                        st.success(f"The {subject_entity} date of birth is {record['o.Name']}.")
                     else:
                         st.warning("No matching subjects found.")

        elif "place of birth" in question:
            with driver.session() as session:
                result = session.run(f"MATCH (s:subject)-[PLACE_OF_BIRTH]->(o) WHERE s.Name = '{subject_entity}' RETURN o.Name")
                for record in result:
                     if record:
                        st.success(f"The {subject_entity} place of birth is {record['o.Name']}.")
                     else:
                         st.warning("No matching subjects found.")

        elif "resides" in question:
            with driver.session() as session:
                result = session.run(f"MATCH (s:subject)-[PLACE_OF_RESIDENCE]->(o) WHERE s.Name = '{subject_entity}' RETURN o.Name")
                for record in result:
                     if record:
                        st.success(f"The {subject_entity} resides at {record['o.Name']}.")
                     else:
                         st.warning("No matching subjects found.")

        elif "employee" in question:
            with driver.session() as session:
                result = session.run(f"MATCH (s:subject)-[EMPLOYEE_OR_MEMBER_OF]->(o) WHERE s.Name = '{subject_entity}' RETURN o.Name")
                for record in result:
                     if record:
                        st.success(f"The {subject_entity}  is an employee of {record['o.Name']}.")
                     else:
                         st.warning("No matching subjects found.")

        # Check if the question is a list query
        if "list" in question:
            if "nationality" in question:
                # Extract the object entity from the question
                object_entity = extract_entity(question, ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"])
                # Run the Neo4j query
                with driver.session() as session:
                    result = session.run(f"MATCH (subject)-[relationship]->(object) WHERE object.Name = '{object_entity}' AND relationship.rel = 'NATIONALITY' RETURN subject")
                    # Get the matching subjects from the result
                    res = [record['subject'].get('Name') for record in result]
                # Display the results to the user
                if res:
                    st.success(f"List of subjects with {object_entity} nationality: {', '.join(res)}")
                else:
                    st.warning("No matching subjects found.")
            
            elif "employee" in question:
                # Extract the object entity from the question
                object_entity = extract_entity(question, ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"])
                # Run the Neo4j query
                with driver.session() as session:
                    result = session.run(f"MATCH (subject)-[relationship]->(object) WHERE object.Name = '{object_entity}' AND relationship.rel = 'EMPLOYEE_OR_MEMBER_OF' RETURN subject")
                    # Get the matching subjects from the result
                    res = [record['subject'].get('Name') for record in result]
                # Display the results to the user
                if res:
                    st.success(f"All employees who works at {object_entity}  : {', '.join(res)}")
                else:
                    st.warning("No matching subjects found.")

            elif "birth" in question:
                # Extract the object entity from the question
                object_entity = extract_entity(question, ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"])
                # Run the Neo4j query
                with driver.session() as session:
                    result = session.run(f"MATCH (subject)-[relationship]->(object) WHERE object.Name = '{object_entity}' AND relationship.rel = 'PLACE_OF_BIRTH' RETURN subject")
                    # Get the matching subjects from the result
                    res = [record['subject'].get('Name') for record in result]
                # Display the results to the user
                if res:
                    st.success(f"All subjects having {object_entity} as place of birth are : {', '.join(res)}")
                else:
                    st.warning("No matching subjects found.")

            elif "resides" in question:
                # Extract the object entity from the question
                object_entity = extract_entity(question, ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"])
                # Run the Neo4j query
                with driver.session() as session:
                    result = session.run(f"MATCH (subject)-[relationship]->(object) WHERE object.Name = '{object_entity}' AND relationship.rel = 'PLACE_OF_RESIDENCE' RETURN subject")
                    # Get the matching subjects from the result
                    res = [record['subject'].get('Name') for record in result]
                # Display the results to the user
                if res:
                    st.success(f"All subjects who resides at {object_entity}  : {', '.join(res)}")
                else:
                    st.warning("No matching subjects found.")

            elif "studied" in question:
                # Extract the object entity from the question
                object_entity = extract_entity(question, ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"])
                # Run the Neo4j query
                with driver.session() as session:
                    result = session.run(f"MATCH (subject)-[relationship]->(object) WHERE object.Name = '{object_entity}' AND relationship.rel = 'EDUCATED_AT' RETURN subject")
                    # Get the matching subjects from the result
                    res = [record['subject'].get('Name') for record in result]
                # Display the results to the user
                if res:
                    st.success(f"All {object_entity} nationalities are : {', '.join(res)}")
                else:
                    st.warning("No matching subjects found.")

            else:
                st.warning("I do not have information related to this list")

        # else:
        #     st.warning("I do not have any information related to this question.")

    # Add a footer to the app
    st.markdown("---")

# Run the app
if __name__ == '__main__':
    app()
    driver.close()
