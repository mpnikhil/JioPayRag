import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import csv
from bs4 import BeautifulSoup

def scrape_faq_with_tailored_selectors(url):
    """
    Scrape FAQ content with selectors tailored to the specific HTML structure.
    """
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to the URL
        print(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for the page to load
        print("Waiting for page to load...")
        time.sleep(10)
        
        # Store all FAQ data
        faq_data = []
        
        # First, identify all section buttons - looking for section titles
        section_elements = driver.find_elements(By.XPATH, 
            "//div[@role='button']/div[@dir='auto' and contains(@class, 'css-1rynq56')]")
        
        print(f"Found {len(section_elements)} section elements")
        
        for section_idx, section_element in enumerate(section_elements):
            try:
                # Get section title
                section_title = section_element.text.strip()
                if not section_title:
                    continue
                    
                print(f"\nProcessing section {section_idx+1}: {section_title}")
                
                # Click to expand section
                parent_button = section_element.find_element(By.XPATH, "./parent::div[@role='button']")
                parent_button.click()
                time.sleep(3)  # Wait for expansion
                
                # Find all questions within this section
                # First, identify the container that holds all questions for this section
                # We need to find a parent container that likely contains all FAQs for this section
                
                # After clicking a section, find all question elements
                # Looking specifically for the structure identified in the sample
                question_elements = driver.find_elements(By.XPATH, 
                    "//div[contains(@class, 'css-1rynq56') and contains(@class, 'r-8jdrp') and contains(@class, 'r-ubezar')]")
                
                qa_pairs = []
                
                for q_idx, question_element in enumerate(question_elements):
                    try:
                        question_text = question_element.text.strip()
                        if not question_text:
                            continue
                            
                        print(f"  Question {q_idx+1}: {question_text[:50]}...")
                        
                        # Find the clickable parent container
                        clickable_parent = question_element.find_element(By.XPATH, 
                            "./ancestor::div[@tabindex='0']")
                        
                        # Click to reveal answer
                        clickable_parent.click()
                        time.sleep(2)  # Wait for answer to appear
                        
                        # The answer should now be visible within a div with max-height style
                        # Get current page source after clicking
                        page_source = driver.page_source
                        soup = BeautifulSoup(page_source, 'html.parser')
                        
                        # Find the question element in the soup
                        question_in_soup = soup.find('div', string=lambda s: s and question_text in s)
                        
                        answer_text = ""
                        if question_in_soup:
                            # Find the tabindex=0 parent
                            tabindex_parent = question_in_soup.find_parent('div', attrs={'tabindex': '0'})
                            
                            if tabindex_parent:
                                # Go up to find the container with max-height style
                                container = tabindex_parent.find_parent('div', class_='css-175oi2r')
                                
                                if container:
                                    # Find the div with max-height style that contains the answer
                                    max_height_div = container.find('div', style=lambda s: s and 'max-height' in s)
                                    
                                    if max_height_div:
                                        # Find all bullet points or text in the answer
                                        bullet_points = max_height_div.find_all('div', 
                                            attrs={'dir': 'auto', 'class': lambda c: c and 'r-8jdrp' in c})
                                        
                                        if bullet_points:
                                            answer_text = '\n'.join([bp.text.strip() for bp in bullet_points])
                        
                        # If we couldn't find the answer with BeautifulSoup, try a direct Selenium approach
                        if not answer_text:
                            try:
                                # Find the div with max-height style
                                max_height_div = driver.find_element(By.XPATH, 
                                    f"//div[contains(@style, 'max-height')]")
                                
                                # Find all text elements within it
                                text_elements = max_height_div.find_elements(By.XPATH, 
                                    ".//div[@dir='auto' and contains(@class, 'r-8jdrp')]")
                                
                                answer_text = '\n'.join([elem.text.strip() for elem in text_elements if elem.text.strip()])
                            except:
                                print("    Couldn't find answer with direct Selenium approach")
                        
                        # Add the QA pair
                        qa_pairs.append({
                            'question': question_text,
                            'answer': answer_text if answer_text else "[No answer found]"
                        })
                        
                        # Click again to collapse (helps with page navigation)
                        try:
                            clickable_parent.click()
                            time.sleep(1)
                        except:
                            pass
                            
                    except Exception as e:
                        print(f"  Error processing question: {str(e)}")
                        continue
                
                # Add data for this section
                if qa_pairs:
                    faq_data.append({
                        'section': section_title,
                        'qa_pairs': qa_pairs
                    })
                
                # Click section again to collapse it
                try:
                    parent_button.click()
                    time.sleep(1)
                except:
                    pass
                    
            except Exception as e:
                print(f"Error processing section: {str(e)}")
                continue
                
        return faq_data
        
    finally:
        driver.quit()

def save_results(faq_data):
    """Save the results to JSON and CSV files"""
    
    # Save to JSON
    with open("faq_data.json", "w", encoding="utf-8") as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)
        
    # Save to CSV
    with open("faq_data.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Question", "Answer"])
        
        for section in faq_data:
            for qa in section['qa_pairs']:
                writer.writerow([
                    section['section'],
                    qa['question'],
                    qa['answer']
                ])
    
    print(f"Saved data to faq_data.json and faq_data.csv")
    
    # Return a summary
    total_sections = len(faq_data)
    total_questions = sum(len(section['qa_pairs']) for section in faq_data)
    return f"Extracted {total_questions} questions from {total_sections} sections"

# Alternative: Process individual HTML files that have been saved after clicking
def extract_qa_from_html_file(html_file):
    """Extract Q&A pairs from a saved HTML file with an expanded answer"""
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    qa_pairs = []
    
    # Find all question elements
    question_elements = soup.find_all('div', class_=lambda c: c and 'r-ubezar' in c and 'r-8jdrp' in c)
    
    for question_element in question_elements:
        question_text = question_element.text.strip()
        if not question_text:
            continue
        
        # Find the tabindex=0 parent
        tabindex_parent = question_element.find_parent('div', attrs={'tabindex': '0'})
        
        if tabindex_parent:
            # Find container with the max-height style
            container = tabindex_parent.find_parent('div', class_='css-175oi2r')
            
            if container:
                max_height_div = container.find('div', style=lambda s: s and 'max-height' in s)
                
                if max_height_div:
                    # Find all bullet points or text in the answer
                    bullet_points = max_height_div.find_all('div', 
                        attrs={'dir': 'auto', 'class': lambda c: c and 'r-8jdrp' in c})
                    
                    answer_text = '\n'.join([bp.text.strip() for bp in bullet_points])
                    
                    qa_pairs.append({
                        'question': question_text,
                        'answer': answer_text if answer_text else "[No answer found]"
                    })
    
    return qa_pairs

if __name__ == "__main__":
    # Option 1: Scrape from URL
    url = "https://jiopay.com/business/help-center"  # Replace with your URL
    data = scrape_faq_with_tailored_selectors(url)
    summary = save_results(data)
    print(summary)
    
    # Option 2: Process a saved HTML file
    # qa_pairs = extract_qa_from_html_file("saved_page.html")
    # print(f"Extracted {len(qa_pairs)} Q&A pairs from the HTML file")