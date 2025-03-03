import time
import json
import os
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException, ElementClickInterceptedException

class JioPayLinkScraper:
    """A scraper for extracting content from JioPay Business website links"""
    
    def __init__(self, headless=False):
        """Initialize the scraper with Chrome options"""
        # Set up Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")  # Updated headless syntax
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        
        # Initialize the Chrome driver
        self.driver = webdriver.Chrome(options=chrome_options)
        self.base_url = "https://jiopay.com/business"
        self.visited_urls = set()
        
    def __del__(self):
        """Clean up when the object is destroyed"""
        try:
            self.driver.quit()
        except:
            pass
    
    def wait_for_page_load(self, timeout=20):
        """Wait for the page to fully load with an extended timeout"""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            # Add additional wait for JavaScript rendering
            time.sleep(5)  # Generous wait for JS to finish rendering
        except TimeoutException:
            print("Timeout waiting for page load")
    
    def extract_text_content(self, element):
        """Extract clean text content from an element"""
        if not element:
            return ""
        
        try:
            # Get all text nodes within the element
            text_elements = element.find_elements(By.XPATH, ".//div[@dir='auto']")
            texts = [elem.text.strip() for elem in text_elements if elem.text.strip()]
            combined_text = "\n".join(texts) if texts else element.text.strip()
            # Clean up the text
            return " ".join(combined_text.split())
        except:
            return element.text.strip() if element else ""
    
    def extract_page_content(self, url, title=None, section=None):
        """Extract content from a page"""
        if url in self.visited_urls:
            print(f"Already visited {url}, skipping...")
            return None
            
        self.visited_urls.add(url)
        print(f"\nExtracting content from: {url}")
        
        try:
            self.driver.get(url)
            # Wait for page to load with a generous timeout
            self.wait_for_page_load(timeout=25)
            
            # Get page title
            if not title:
                try:
                    title_element = self.driver.find_element(By.XPATH, 
                        "//div[contains(@class, 'r-1xnzce8') and contains(@class, 'r-b4mj7b')]")
                    title = title_element.text.strip()
                except:
                    title = self.driver.title
            
            print(f"Page title: {title}")
            
            # Extract content sections
            content_sections = []
            section_elements = self.driver.find_elements(By.XPATH, 
                "//div[contains(@class, 'css-175oi2r') and ./div[@dir='auto']]")
            
            for idx, section_elem in enumerate(section_elements):
                # Skip very small sections or empty sections
                try:
                    if section_elem.size['height'] < 30 or section_elem.size['width'] < 200:
                        continue
                except:
                    continue
                
                try:
                    # Extract section title
                    section_title = ""
                    try:
                        section_title_elem = section_elem.find_element(By.XPATH, 
                            ".//div[contains(@class, 'r-1xnzce8') or contains(@class, 'r-ubezar')]")
                        section_title = section_title_elem.text.strip()
                    except:
                        pass
                    
                    # Extract section content
                    section_content = self.extract_text_content(section_elem)
                    
                    # Only add if section has meaningful content
                    if section_content and len(section_content) > 20 and section_content != section_title:
                        content_sections.append({
                            "title": section_title,
                            "content": section_content
                        })
                except Exception as e:
                    print(f"Error extracting section {idx}: {str(e)}")
            
            # Extract any lists or bullet points
            lists = []
            try:
                list_containers = self.driver.find_elements(By.XPATH, 
                    "//div[contains(@class, 'r-v7kwyn') or contains(@class, 'r-1awozwy')]")
                
                for container in list_containers:
                    try:
                        list_items = container.find_elements(By.XPATH, 
                            ".//div[@dir='auto' and contains(@class, 'r-8jdrp')]")
                        
                        items = [item.text.strip() for item in list_items if item.text.strip()]
                        if len(items) >= 2:  # Only consider it a list if it has at least 2 items
                            lists.append(items)
                    except:
                        continue
            except:
                pass
            
            # Removed image extraction as requested
            
            # Extract any buttons or CTA elements
            cta_elements = []
            try:
                buttons = self.driver.find_elements(By.XPATH, 
                    "//div[@role='button' and contains(@class, 'r-lrvibr')]")
                
                for button in buttons:
                    try:
                        button_text = button.text.strip()
                        if button_text:
                            cta_elements.append(button_text)
                    except:
                        continue
            except:
                pass
            
            # Extract FAQs if this page appears to contain them
            faqs = self.extract_faqs_from_page()
            
            # Create structured page data
            page_data = {
                "url": url,
                "title": title,
                "section": section,
                "content_sections": content_sections,
                "lists": lists,
                "cta_elements": cta_elements,
                "faqs": faqs
            }
            
            return page_data
            
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return {
                "url": url,
                "title": title or "Error",
                "section": section,
                "content_sections": [],
                "lists": [],
                "cta_elements": [],
                "faqs": [],
                "error": str(e)
            }
    
    def click_link_by_text(self, link_text, max_retries=3):
        """Click a link by its text with retries and generous waits"""
        print(f"Attempting to click link: {link_text}")
        
        for attempt in range(max_retries):
            try:
                # Find elements that could contain the link text
                elements = self.driver.find_elements(By.XPATH, 
                    f"//div[contains(text(), '{link_text}') or ./div[contains(text(), '{link_text}')]]")
                
                # Try to find the most likely link element
                for elem in elements:
                    if link_text in elem.text:
                        # Check if it's clickable (has a tabindex or role attribute)
                        if elem.get_attribute("tabindex") or elem.get_attribute("role") == "button":
                            print(f"Found clickable element for: {link_text}")
                            # Scroll to element to ensure it's visible
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
                            time.sleep(2)  # Wait after scrolling
                            elem.click()
                            time.sleep(5)  # Generous wait after clicking
                            return True
                        
                        # Check if it has a clickable parent
                        try:
                            parent = elem.find_element(By.XPATH, "./ancestor::div[@role='button' or @tabindex='0'][1]")
                            print(f"Found clickable parent for: {link_text}")
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", parent)
                            time.sleep(2)
                            parent.click()
                            time.sleep(5)
                            return True
                        except:
                            pass
                
                # If we haven't found a match yet, try a different approach
                try:
                    # Look for elements with the exact text
                    exact_match = self.driver.find_element(By.XPATH, 
                        f"//div[@dir='auto' and text()='{link_text}']")
                    print(f"Found exact match for: {link_text}")
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", exact_match)
                    time.sleep(2)
                    exact_match.click()
                    time.sleep(5)
                    return True
                except:
                    pass
                
                print(f"Attempt {attempt+1}/{max_retries} failed for link: {link_text}")
                # Wait before retry with a random component
                time.sleep(2 + random.random() * 2)
                
            except (StaleElementReferenceException, ElementClickInterceptedException) as e:
                print(f"Error clicking link (attempt {attempt+1}): {str(e)}")
                time.sleep(3)  # Wait before retry
        
        print(f"Failed to click link after {max_retries} attempts: {link_text}")
        return False
    
    def scrape_links_from_footer(self):
        """Scrape content from all links in the footer"""
        print("Starting to scrape links from footer...")
        
        # First, navigate to the home page and extract footer links
        self.driver.get(self.base_url)
        self.wait_for_page_load(timeout=25)
        
        # Extract footer categories and links
        footer_data = self.extract_footer_structure()
        
        # Data structure to store all the scraped content
        scraped_data = {
            "home_page": self.extract_page_content(self.base_url, "Home"),
            "categories": []
        }
        
        # Process each category and its links
        for category, links in footer_data.items():
            print(f"\n--- Processing category: {category} ---")
            category_data = {
                "name": category,
                "links": []
            }
            
            for link_text in links:
                # Navigate back to home page for each new link
                self.driver.get(self.base_url)
                self.wait_for_page_load(timeout=25)
                
                # Try to click the link
                if self.click_link_by_text(link_text):
                    # Wait for the page to load
                    self.wait_for_page_load(timeout=25)
                    
                    # Get current URL
                    current_url = self.driver.current_url
                    
                    # Extract page content
                    page_data = self.extract_page_content(current_url, link_text, category)
                    
                    if page_data:
                        category_data["links"].append(page_data)
                else:
                    print(f"Skipping {link_text} - could not click the link")
                    category_data["links"].append({
                        "title": link_text,
                        "error": "Could not navigate to page"
                    })
                
                # Add a delay between processing links
                time.sleep(3 + random.random() * 2)
            
            scraped_data["categories"].append(category_data)
        
        return scraped_data
    
    def extract_faqs_from_page(self):
        """Extract FAQs from the current page"""
        faqs = []
        
        # Look for possible FAQ section headers
        try:
            # Method 1: Look for section elements with role='button'
            section_elements = self.driver.find_elements(By.XPATH, 
                "//div[@role='button']/div[contains(@class, 'css-1rynq56')]")
            
            if section_elements:
                for section_element in section_elements:
                    try:
                        section_title = section_element.text.strip()
                        if not section_title:
                            continue
                        
                        print(f"Found potential FAQ section: {section_title}")
                        
                        # Try to click to expand section
                        try:
                            parent_button = section_element.find_element(By.XPATH, "./parent::div[@role='button']")
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", parent_button)
                            time.sleep(1)
                            parent_button.click()
                            time.sleep(3)  # Wait for expansion
                        except:
                            print("Failed to expand section")
                            continue
                        
                        # Process questions in this section
                        qa_pairs = self.process_faq_questions()
                        
                        if qa_pairs:
                            faqs.append({
                                "section": section_title,
                                "qa_pairs": qa_pairs
                            })
                        
                        # Try to collapse section
                        try:
                            parent_button.click()
                            time.sleep(1)
                        except:
                            pass
                    except Exception as e:
                        print(f"Error processing FAQ section: {str(e)}")
            
            # Method 2: Look for FAQ-like questions that don't have section headers
            if not faqs:
                # Look for elements that look like questions
                question_elements = self.driver.find_elements(By.XPATH, 
                    "//div[contains(@class, 'r-lrvibr')]//div[@dir='auto' and contains(@class, 'r-1rtz1e9')]")
                
                if question_elements:
                    qa_pairs = self.process_faq_questions(question_elements)
                    if qa_pairs:
                        faqs.append({
                            "section": "General FAQs",
                            "qa_pairs": qa_pairs
                        })
            
        except Exception as e:
            print(f"Error extracting FAQs: {str(e)}")
        
        return faqs
    
    def process_faq_questions(self, question_elements=None):
        """Process FAQ questions and extract answers"""
        qa_pairs = []
        
        try:
            # If no question elements provided, try to find them
            if not question_elements:
                # This expanded XPath captures more question patterns, including the specific one mentioned
                question_elements = self.driver.find_elements(By.XPATH, 
                    "//div[contains(@class, 'r-ubezar') and contains(@class, 'r-8jdrp') or " +
                    "contains(@class, 'r-1rtz1e9') and contains(@class, 'r-1b43r93') or " +
                    "contains(@class, 'r-op4f77') and contains(@class, 'r-1rtz1e9')]")
            
            for question_element in question_elements:
                try:
                    question_text = question_element.text.strip()
                    if not question_text or len(question_text) < 5:  # Skip very short texts
                        continue
                    
                    print(f"Processing FAQ question: {question_text[:50]}...")
                    
                    # Find the clickable container
                    clickable_parent = None
                    try:
                        # Try several patterns to find the clickable parent
                        for xpath in [
                            "./ancestor::div[@tabindex='0' or @role='button'][1]",
                            "./parent::div[contains(@class, 'r-lrvibr')]",
                            "./ancestor::div[contains(@class, 'r-lrvibr')][1]",
                            "./ancestor::div[@tabindex='0'][1]"  # Added for the specific pattern
                        ]:
                            try:
                                clickable_parent = question_element.find_element(By.XPATH, xpath)
                                if clickable_parent:
                                    break
                            except:
                                continue
                    except:
                        print(f"Could not find clickable parent for question")
                        continue
                    
                    if not clickable_parent:
                        continue
                    
                    # Scroll to the question element
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", clickable_parent)
                    time.sleep(1)
                    
                    # Check if answer is already visible before clicking (for static FAQs)
                    answer_before_click = self.check_for_visible_answer(question_text)
                    if answer_before_click:
                        qa_pairs.append({
                            "question": question_text,
                            "answer": answer_before_click
                        })
                        print(f"Found answer without clicking: {answer_before_click[:50]}...")
                        continue
                    
                    # Click to reveal answer
                    try:
                        clickable_parent.click()
                        time.sleep(2)  # Wait for answer to appear
                    except:
                        print(f"Failed to click question element")
                        continue
                    
                    # Try multiple approaches to find the answer
                    answer_text = ""
                    
                    # Approach 1: Look for elements with r-9daio6 class (specific to the example structure)
                    try:
                        answer_elements = self.driver.find_elements(By.XPATH, 
                            "//div[contains(@class, 'r-9daio6') and contains(@class, 'r-8jdrp') and @dir='auto']")
                        
                        # Filter for likely answers
                        for elem in answer_elements:
                            if elem.is_displayed() and elem.text.strip() and elem.text.strip() != question_text:
                                answer_text = elem.text.strip()
                                break
                    except:
                        pass
                    
                    # Approach 2: Look for expanded content div
                    if not answer_text:
                        try:
                            # Look for elements within max-height divs
                            answer_elements = self.driver.find_elements(By.XPATH, 
                                "//div[contains(@style, 'max-height')]//div[@dir='auto']")
                            
                            # Combine all text from answer elements
                            answer_texts = [elem.text.strip() for elem in answer_elements 
                                           if elem.is_displayed() and elem.text.strip() and elem.text.strip() != question_text]
                            if answer_texts:
                                answer_text = "\n".join(answer_texts)
                        except:
                            pass
                    
                    # Approach 3: Look for specific structure with r-95jzfe class
                    if not answer_text:
                        try:
                            # This is specifically targeting the structure in your example
                            answer_element = clickable_parent.find_element(By.XPATH, 
                                "./following-sibling::div[contains(@class, 'r-95jzfe')]//div[contains(@class, 'r-9daio6')]")
                            
                            if answer_element.is_displayed():
                                answer_text = answer_element.text.strip()
                        except:
                            pass
                    
                    # Approach 4: Look for answer by checking sibling elements
                    if not answer_text:
                        try:
                            siblings = self.driver.find_elements(By.XPATH, 
                                f"//div[contains(text(), '{question_text}')]/following::div[2]//div[@dir='auto']")
                            
                            for sibling in siblings:
                                if sibling.is_displayed() and sibling.text.strip() and sibling.text.strip() != question_text:
                                    answer_text = sibling.text.strip()
                                    break
                        except:
                            pass
                    
                    # If we found an answer, add it to our data
                    if answer_text and answer_text != question_text:
                        qa_pairs.append({
                            "question": question_text,
                            "answer": answer_text
                        })
                        print(f"Found answer: {answer_text[:50]}...")
                    else:
                        print(f"No answer found for question")
                    
                    # Click again to collapse
                    try:
                        clickable_parent.click()
                        time.sleep(1)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"Error processing question: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing FAQ questions: {str(e)}")
        
        return qa_pairs
    
    def check_for_visible_answer(self, question_text):
        """Check if an answer is already visible without clicking (for static FAQs)"""
        try:
            # Try to find answer elements that are siblings to the question
            answer_elements = self.driver.find_elements(By.XPATH, 
                f"//div[contains(text(), '{question_text}')]/following-sibling::div//div[@dir='auto']")
            
            for elem in answer_elements:
                if elem.is_displayed() and elem.text.strip() and elem.text.strip() != question_text:
                    return elem.text.strip()
            
            return None
        except:
            return None
    
    def extract_footer_structure(self):
        """Extract the structure of the footer links"""
        footer_links = {}
        
        try:
            # Find the footer
            footer = self.driver.find_element(By.XPATH, "//div[@id='footer']")
            
            # Find categories
            categories = footer.find_elements(By.XPATH, 
                ".//div[.//div[contains(@class, 'r-ubezar')]]")
            
            for category in categories:
                try:
                    # Get category name
                    category_name_elem = category.find_element(By.XPATH, 
                        ".//div[contains(@class, 'r-ubezar')]")
                    category_name = category_name_elem.text.strip()
                    
                    if not category_name:
                        continue
                    
                    # Get links
                    link_elements = category.find_elements(By.XPATH, 
                        ".//div[@tabindex='0']//div[contains(@class, 'r-8jdrp')]")
                    
                    links = [link.text.strip() for link in link_elements if link.text.strip()]
                    
                    if links:
                        footer_links[category_name] = links
                except Exception as e:
                    print(f"Error extracting category: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error extracting footer structure: {str(e)}")
        
        return footer_links
    
    def save_results(self, data, output_dir="."):
        """Save the scraped data to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save to JSON
        json_path = os.path.join(output_dir, "jiopay_links_content.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Create a simplified CSV version for general content
        import csv
        csv_path = os.path.join(output_dir, "jiopay_links_content.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Link Title", "URL", "Section Title", "Content"])
            
            # Process home page
            if "home_page" in data and data["home_page"]:
                for section in data["home_page"].get("content_sections", []):
                    writer.writerow([
                        "Home",
                        data["home_page"].get("title", "Home"),
                        data["home_page"].get("url", ""),
                        section.get("title", ""),
                        section.get("content", "")
                    ])
            
            # Process categories
            for category in data.get("categories", []):
                category_name = category.get("name", "")
                
                for link in category.get("links", []):
                    # Skip links with errors
                    if "error" in link and not link.get("content_sections"):
                        continue
                        
                    for section in link.get("content_sections", []):
                        writer.writerow([
                            category_name,
                            link.get("title", ""),
                            link.get("url", ""),
                            section.get("title", ""),
                            section.get("content", "")
                        ])
        
        # Create a separate CSV for FAQs
        faq_csv_path = os.path.join(output_dir, "jiopay_faqs.csv")
        with open(faq_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Source", "Section", "Question", "Answer"])
            
            # Process FAQs from homepage
            if "home_page" in data and data["home_page"] and data["home_page"].get("faqs"):
                for faq_section in data["home_page"].get("faqs", []):
                    section_name = faq_section.get("section", "")
                    for qa in faq_section.get("qa_pairs", []):
                        writer.writerow([
                            "Home",
                            section_name,
                            qa.get("question", ""),
                            qa.get("answer", "")
                        ])
            
            # Process FAQs from all other pages
            for category in data.get("categories", []):
                category_name = category.get("name", "")
                
                for link in category.get("links", []):
                    link_title = link.get("title", "")
                    
                    for faq_section in link.get("faqs", []):
                        section_name = faq_section.get("section", "")
                        for qa in faq_section.get("qa_pairs", []):
                            writer.writerow([
                                f"{category_name} - {link_title}",
                                section_name,
                                qa.get("question", ""),
                                qa.get("answer", "")
                            ])
        
        # Count FAQs
        total_faqs = 0
        # From homepage
        if "home_page" in data and data["home_page"] and data["home_page"].get("faqs"):
            for section in data["home_page"].get("faqs", []):
                total_faqs += len(section.get("qa_pairs", []))
        
        # From all other pages
        for category in data.get("categories", []):
            for link in category.get("links", []):
                for section in link.get("faqs", []):
                    total_faqs += len(section.get("qa_pairs", []))
        
        print(f"Data saved to {json_path} and {csv_path}")
        print(f"FAQs saved to {faq_csv_path} (Total FAQs: {total_faqs})")
        
        return {
            "json_path": json_path,
            "csv_path": csv_path,
            "faq_csv_path": faq_csv_path,
            "total_faqs": total_faqs
        }

def run_link_scraper(headless=False, output_dir="jiopay_links_data"):
    """Run the JioPay link scraper"""
    scraper = JioPayLinkScraper(headless=headless)
    try:
        print("Starting JioPay link scraper...")
        data = scraper.scrape_links_from_footer()
        return scraper.save_results(data, output_dir=output_dir)
    finally:
        scraper.__del__()

if __name__ == "__main__":
    # Run the link scraper with specified options
    result = run_link_scraper(
        headless=False,  # Set to True for headless mode
        output_dir="jiopay_links_data"
    )
    print(f"Scraping completed. Files saved: {result}")