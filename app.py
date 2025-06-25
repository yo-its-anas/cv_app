import streamlit as st
import pandas as pd
import pypdf
import io
import spacy
from spacy.matcher import PhraseMatcher
from datetime import datetime
import re
import base64

# Load English language model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please install the spaCy English model by running: python -m spacy download en_core_web_sm")
    st.stop()

# Predefined lists of skills and degree types
TECH_SKILLS = [
    'python', 'sql', 'excel', 'aws', 'machine learning', 'deep learning', 
    'tensorflow', 'pytorch', 'scikit-learn', 'data analysis', 'power bi', 
    'tableau', 'java', 'c++', 'javascript', 'html', 'css', 'react', 'angular', 
    'node.js', 'docker', 'kubernetes', 'git', 'linux', 'nosql', 'mongodb', 
    'postgresql', 'mysql', 'spark', 'hadoop', 'pandas', 'numpy'
]

SOFT_SKILLS = [
    'communication', 'teamwork', 'leadership', 'problem solving', 
    'time management', 'adaptability', 'creativity', 'critical thinking', 
    'emotional intelligence', 'collaboration', 'negotiation', 
    'conflict resolution', 'decision making'
]

DEGREE_TYPES = [
    'bachelor', 'master', 'phd', 'doctorate', 'bs', 'ms', 'mba', 'bsc', 
    'btech', 'mtech', 'ba', 'ma'
]

# Initialize session state for storing CV data
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = pd.DataFrame(columns=[
        'Name', 'Email', 'Phone', 'Total_Experience', 'Skills', 
        'Education', 'Previous_Roles', 'Companies', 'Achievements', 
        'Raw_Text', 'Score', 'File_Name'
    ])

def extract_text_from_pdf(uploaded_file):
    """Extract raw text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_contact_info(text):
    """Extract name, email, and phone number from text"""
    doc = nlp(text)
    
    # Extract name (first entity recognized as PERSON)
    name = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    
    # Extract email using regex
    email = ""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        email = email_match.group()
    
    # Extract phone number using regex
    phone = ""
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        phone = phone_match.group()
    
    return name, email, phone

def extract_experience(text):
    """Estimate total years of experience from text"""
    # Look for patterns like "5 years of experience" or "experience: 2010-2015"
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*:\s*(\d+)\+?\s*years?',
        r'(\d+)\s*yr',
        r'(\d+)\s*yrs'
    ]
    
    max_experience = 0
    for pattern in experience_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            years = int(match.group(1))
            if years > max_experience:
                max_experience = years
    
    # Fallback: look for date ranges in the text
    if max_experience == 0:
        date_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|present|current)'
        matches = re.finditer(date_pattern, text, re.IGNORECASE)
        date_ranges = []
        for match in matches:
            start_year = int(match.group(1))
            end_year = match.group(2).lower()
            if end_year in ['present', 'current']:
                end_year = datetime.now().year
            else:
                end_year = int(end_year)
            date_ranges.append((start_year, end_year))
        
        if date_ranges:
            # Calculate total experience from date ranges
            total_months = 0
            for start, end in date_ranges:
                total_months += (end - start) * 12
            max_experience = round(total_months / 12)
    
    return max_experience

def extract_skills(text):
    """Extract technical and soft skills from text"""
    doc = nlp(text.lower())
    
    # Create PhraseMatcher for skills
    tech_matcher = PhraseMatcher(nlp.vocab)
    soft_matcher = PhraseMatcher(nlp.vocab)
    
    tech_patterns = [nlp(skill) for skill in TECH_SKILLS]
    soft_patterns = [nlp(skill) for skill in SOFT_SKILLS]
    
    tech_matcher.add("TECH_SKILLS", tech_patterns)
    soft_matcher.add("SOFT_SKILLS", soft_patterns)
    
    tech_matches = tech_matcher(doc)
    soft_matches = soft_matcher(doc)
    
    tech_skills = set()
    soft_skills = set()
    
    for match_id, start, end in tech_matches:
        span = doc[start:end]
        tech_skills.add(span.text)
    
    for match_id, start, end in soft_matches:
        span = doc[start:end]
        soft_skills.add(span.text)
    
    return list(tech_skills), list(soft_skills)

def extract_education(text):
    """Extract education information"""
    doc = nlp(text)
    education = []
    
    # Look for degree types followed by institution names
    for sent in doc.sents:
        for degree in DEGREE_TYPES:
            if degree in sent.text.lower():
                education.append(sent.text)
                break
    
    return education

def extract_previous_roles(text):
    """Extract previous job titles"""
    doc = nlp(text)
    roles = []
    
    # Look for common job title patterns
    for ent in doc.ents:
        if ent.label_ == "ORG":
            # Check if preceding words might be a title
            start = max(0, ent.start - 3)
            prefix = doc[start:ent.start]
            if any(word.lower() in ['at', 'of', 'from'] for word in prefix):
                roles.append(prefix.text + " " + ent.text)
    
    # Alternative approach: look for lines with dates followed by titles
    date_pattern = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\b|\b\d{4}\s*[-–]\s*\d{4}\b'
    lines = text.split('\n')
    for line in lines:
        if re.search(date_pattern, line, re.IGNORECASE):
            # Remove dates and company names
            clean_line = re.sub(date_pattern, '', line)
            clean_line = re.sub(r'@\w+', '', clean_line)
            if clean_line.strip():
                roles.append(clean_line.strip())
    
    return list(set(roles))[:5]  # Return up to 5 unique roles

def extract_companies(text):
    """Extract company names"""
    doc = nlp(text)
    companies = set()
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            companies.add(ent.text)
    
    return list(companies)[:5]  # Return up to 5 companies

def extract_achievements(text):
    """Extract quantifiable achievements"""
    # Look for patterns with numbers and action verbs
    achievement_patterns = [
        r'increased\s.*?\b\d+%',
        r'reduced\s.*?\b\d+%',
        r'saved\s.*?\b\d+%',
        r'achieved\s.*?\b\d+',
        r'improved\s.*?\b\d+%',
        r'led\s.*?\bteam\s.*?\b\d+',
        r'managed\s.*?\bbudget\s.*?\b\$\d+'
    ]
    
    achievements = []
    for pattern in achievement_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            achievements.append(match.group())
    
    return achievements[:3]  # Return up to 3 achievements

def calculate_score(data, rules):
    """Calculate score based on custom rules"""
    score = 0
    
    # Years of experience rules
    if 'years_experience' in rules:
        for condition, points in rules['years_experience'].items():
            if condition.startswith('>='):
                min_years = int(condition[2:])
                if data['Total_Experience'] >= min_years:
                    score += points
            elif condition.startswith('>'):
                min_years = int(condition[1:])
                if data['Total_Experience'] > min_years:
                    score += points
    
    # Skills rules
    if 'skills' in rules:
        all_skills = data['Skills'][0] + data['Skills'][1]  # Tech + soft skills
        for skill, points in rules['skills'].items():
            if skill.lower() in [s.lower() for s in all_skills]:
                score += points
    
    # Education rules
    if 'education' in rules:
        for degree, points in rules['education'].items():
            if any(degree.lower() in edu.lower() for edu in data['Education']):
                score += points
    
    # Previous roles rules
    if 'previous_roles' in rules:
        for role_pattern, points in rules['previous_roles'].items():
            if any(role_pattern.lower() in role.lower() for role in data['Previous_Roles']):
                score += points
    
    # Negative scoring rules
    if 'negative' in rules:
        for condition, penalty in rules['negative'].items():
            if condition.startswith('skill_'):
                required_skill = condition[6:]
                if required_skill.lower() not in [s.lower() for s in data['Skills'][0]]:
                    score -= penalty
    
    return score

def process_cv(uploaded_file, rules):
    """Process a single CV file"""
    file_name = uploaded_file.name
    text = extract_text_from_pdf(uploaded_file)
    if not text:
        return None
    
    name, email, phone = extract_contact_info(text)
    experience = extract_experience(text)
    tech_skills, soft_skills = extract_skills(text)
    education = extract_education(text)
    previous_roles = extract_previous_roles(text)
    companies = extract_companies(text)
    achievements = extract_achievements(text)
    
    data = {
        'Name': name,
        'Email': email,
        'Phone': phone,
        'Total_Experience': experience,
        'Skills': (tech_skills, soft_skills),
        'Education': education,
        'Previous_Roles': previous_roles,
        'Companies': companies,
        'Achievements': achievements,
        'Raw_Text': text,
        'Score': 0,
        'File_Name': file_name
    }
    
    data['Score'] = calculate_score(data, rules)
    
    return data

def display_cv_data(df):
    """Display CV data in an interactive table"""
    st.subheader("Candidate Summary")
    
    # Create a display DataFrame with selected columns
    display_df = df.copy()
    display_df['Tech Skills'] = display_df['Skills'].apply(lambda x: ', '.join(x[0][:3]) + ('...' if len(x[0]) > 3 else ''))
    display_df['Soft Skills'] = display_df['Skills'].apply(lambda x: ', '.join(x[1][:3]) + ('...' if len(x[1]) > 3 else ''))
    display_df['Education'] = display_df['Education'].apply(lambda x: ', '.join(x[:2]) + ('...' if len(x) > 2 else ''))
    
    columns_to_display = [
        'Name', 'Score', 'Total_Experience', 'Tech Skills', 'Soft Skills', 
        'Education', 'File_Name'
    ]
    
    st.dataframe(
        display_df[columns_to_display],
        column_config={
            "Name": "Name",
            "Score": st.column_config.NumberColumn(
                "Score",
                help="Candidate score based on defined rules",
                format="%d"
            ),
            "Total_Experience": st.column_config.NumberColumn(
                "Years Exp",
                help="Total years of experience",
                format="%d"
            ),
            "Tech Skills": "Technical Skills",
            "Soft Skills": "Soft Skills",
            "Education": "Education",
            "File_Name": "CV File"
        },
        hide_index=True,
        use_container_width=True
    )

def get_table_download_link(df):
    """Generates a link allowing the data in the given dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="candidate_data.csv">Download CSV</a>'
    return href

def main():
    st.title("AI-Powered CV Filtering System")
    st.markdown("Upload candidate CVs in PDF format to automatically extract information and score candidates based on your criteria.")
    
    # Initialize default scoring rules
    default_rules = {
        'years_experience': {
            '>=1': 10,
            '>=3': 30,
            '>=5': 50,
            '>=10': 80
        },
        'skills': {
            'python': 20,
            'sql': 15,
            'excel': 10,
            'aws': 20,
            'machine learning': 25,
            'communication': 15,
            'leadership': 20
        },
        'education': {
            'bachelor': 20,
            'master': 30,
            'phd': 40,
            'mba': 25
        },
        'previous_roles': {
            'manager': 30,
            'director': 40,
            'vp': 50,
            'engineer': 20,
            'analyst': 15
        },
        'negative': {
            'skill_python': 30  # Penalty if Python is not present
        }
    }
    
    # Rule configuration in sidebar
    st.sidebar.header("Scoring Rules Configuration")
    
    with st.sidebar.expander("Experience Rules"):
        exp_1 = st.number_input("1+ years", value=10, key="exp_1")
        exp_3 = st.number_input("3+ years", value=30, key="exp_3")
        exp_5 = st.number_input("5+ years", value=50, key="exp_5")
        exp_10 = st.number_input("10+ years", value=80, key="exp_10")
    
    with st.sidebar.expander("Technical Skills"):
        python = st.number_input("Python", value=20, key="python")
        sql = st.number_input("SQL", value=15, key="sql")
        excel = st.number_input("Excel", value=10, key="excel")
        aws = st.number_input("AWS", value=20, key="aws")
        ml = st.number_input("Machine Learning", value=25, key="ml")
    
    with st.sidebar.expander("Soft Skills"):
        communication = st.number_input("Communication", value=15, key="communication")
        leadership = st.number_input("Leadership", value=20, key="leadership")
    
    with st.sidebar.expander("Education"):
        bachelor = st.number_input("Bachelor's", value=20, key="bachelor")
        master = st.number_input("Master's", value=30, key="master")
        phd = st.number_input("PhD", value=40, key="phd")
        mba = st.number_input("MBA", value=25, key="mba")
    
    with st.sidebar.expander("Previous Roles"):
        manager = st.number_input("Manager", value=30, key="manager")
        director = st.number_input("Director", value=40, key="director")
        vp = st.number_input("VP", value=50, key="vp")
        engineer = st.number_input("Engineer", value=20, key="engineer")
        analyst = st.number_input("Analyst", value=15, key="analyst")
    
    with st.sidebar.expander("Negative Rules"):
        python_penalty = st.number_input("Penalty if no Python", value=30, key="python_penalty")
    
    # Update rules based on user input
    rules = {
        'years_experience': {
            '>=1': exp_1,
            '>=3': exp_3,
            '>=5': exp_5,
            '>=10': exp_10
        },
        'skills': {
            'python': python,
            'sql': sql,
            'excel': excel,
            'aws': aws,
            'machine learning': ml,
            'communication': communication,
            'leadership': leadership
        },
        'education': {
            'bachelor': bachelor,
            'master': master,
            'phd': phd,
            'mba': mba
        },
        'previous_roles': {
            'manager': manager,
            'director': director,
            'vp': vp,
            'engineer': engineer,
            'analyst': analyst
        },
        'negative': {
            'skill_python': python_penalty
        }
    }
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload CVs (PDF format)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process CVs"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                cv_data = process_cv(uploaded_file, rules)
                if cv_data:
                    # Convert to DataFrame and append to session state
                    new_row = pd.DataFrame([cv_data])
                    st.session_state.cv_data = pd.concat(
                        [st.session_state.cv_data, new_row], 
                        ignore_index=True
                    )
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            st.balloons()
    
    # Display and filter data if we have any
    if not st.session_state.cv_data.empty:
        st.divider()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider(
                "Minimum Score", 
                min_value=0, 
                max_value=int(st.session_state.cv_data['Score'].max()), 
                value=50
            )
        with col2:
            min_experience = st.slider(
                "Minimum Years Experience", 
                min_value=0, 
                max_value=int(st.session_state.cv_data['Total_Experience'].max()), 
                value=2
            )
        with col3:
            required_skill = st.selectbox(
                "Filter by Skill", 
                options=["All"] + TECH_SKILLS + SOFT_SKILLS
            )
        
        # Apply filters
        filtered_df = st.session_state.cv_data[
            (st.session_state.cv_data['Score'] >= min_score) &
            (st.session_state.cv_data['Total_Experience'] >= min_experience)
        ].copy()
        
        if required_skill != "All":
            filtered_df = filtered_df[
                filtered_df['Skills'].apply(
                    lambda x: required_skill.lower() in [s.lower() for s in x[0] + x[1]]
                )
            ]
        
        # Display filtered results
        if not filtered_df.empty:
            display_cv_data(filtered_df)
            
            # Sorting options
            sort_by = st.selectbox(
                "Sort By",
                options=["Score (High to Low)", "Score (Low to High)", 
                        "Experience (High to Low)", "Experience (Low to High)"]
            )
            
            if sort_by == "Score (High to Low)":
                filtered_df = filtered_df.sort_values('Score', ascending=False)
            elif sort_by == "Score (Low to High)":
                filtered_df = filtered_df.sort_values('Score', ascending=True)
            elif sort_by == "Experience (High to Low)":
                filtered_df = filtered_df.sort_values('Total_Experience', ascending=False)
            else:
                filtered_df = filtered_df.sort_values('Total_Experience', ascending=True)
            
            # Display detailed view when a candidate is selected
            selected_name = st.selectbox(
                "View Candidate Details",
                options=["Select a candidate"] + filtered_df['Name'].tolist()
            )
            
            if selected_name != "Select a candidate":
                selected_candidate = filtered_df[filtered_df['Name'] == selected_name].iloc[0]
                
                st.subheader(f"Detailed View: {selected_candidate['Name']}")
                st.write(f"**Score:** {selected_candidate['Score']}")
                st.write(f"**Years of Experience:** {selected_candidate['Total_Experience']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Contact Information**")
                    st.write(f"Email: {selected_candidate['Email']}")
                    st.write(f"Phone: {selected_candidate['Phone']}")
                    
                    st.write("**Education**")
                    for edu in selected_candidate['Education']:
                        st.write(f"- {edu}")
                
                with col2:
                    st.write("**Technical Skills**")
                    for skill in selected_candidate['Skills'][0]:
                        st.write(f"- {skill}")
                    
                    st.write("**Soft Skills**")
                    for skill in selected_candidate['Skills'][1]:
                        st.write(f"- {skill}")
                
                st.write("**Previous Roles**")
                for role in selected_candidate['Previous_Roles']:
                    st.write(f"- {role}")
                
                st.write("**Companies**")
                for company in selected_candidate['Companies']:
                    st.write(f"- {company}")
                
                if selected_candidate['Achievements']:
                    st.write("**Key Achievements**")
                    for achievement in selected_candidate['Achievements']:
                        st.write(f"- {achievement}")
                
                st.download_button(
                    label="Download Full CV Text",
                    data=selected_candidate['Raw_Text'],
                    file_name=f"{selected_candidate['Name']}_cv_text.txt",
                    mime="text/plain"
                )
            
            # Download all data as CSV
            st.markdown(get_table_download_link(filtered_df), unsafe_allow_html=True)
        else:
            st.warning("No candidates match the current filters.")
    else:
        st.info("Upload CVs to begin processing.")

if __name__ == "__main__":
    main()
