# Make sure the file is named 'improved_jiopay_scraper.py'
from improved_jiopay_scraper import JioPayLinkScraper
import json
import os
import time

def main():
    """Run the improved JioPay scraper with both general and FAQ-specific scraping"""
    output_dir = "jiopay_data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the scraper
    scraper = JioPayLinkScraper(headless=False)  # Set to True for headless mode in production
    
    try:
        # First, do a direct scrape of the help center to ensure we get all FAQs
        print("="*80)
        print("STEP 1: Direct scrape of the Help Center FAQ page")
        print("="*80)
        faq_data = scraper.run_direct_faq_scrape()
        
        if faq_data:
            # Save the FAQ data separately
            faq_path = os.path.join(output_dir, "jiopay_help_center_faqs.json")
            with open(faq_path, "w", encoding="utf-8") as f:
                json.dump(faq_data, f, ensure_ascii=False, indent=2)
            print(f"Help Center FAQs saved to {faq_path}")
        
        # Give the browser a rest between operations
        time.sleep(5)
        
        # Now, do the general site scraping
        print("\n")
        print("="*80)
        print("STEP 2: General site scraping including footer links")
        print("="*80)
        general_data = scraper.scrape_links_from_footer()
        
        # Save the general scraping results
        result_info = scraper.save_results(general_data, output_dir=output_dir)
        
        print("\n")
        print("="*80)
        print("SCRAPING COMPLETED")
        print("="*80)
        print(f"General content saved to: {result_info['json_path']}")
        print(f"Content CSV saved to: {result_info['csv_path']}")
        print(f"FAQs CSV saved to: {result_info['faq_csv_path']}")
        print(f"Total FAQs extracted from general scrape: {result_info['total_faqs']}")
        
        if faq_data:
            help_center_faqs = sum(len(section.get('qa_pairs', [])) for section in faq_data)
            print(f"Additional FAQs from Help Center direct scrape: {help_center_faqs}")
            print(f"Grand total FAQs: {result_info['total_faqs'] + help_center_faqs}")
        
    finally:
        # Ensure the driver is closed properly
        scraper.__del__()

if __name__ == "__main__":
    main()