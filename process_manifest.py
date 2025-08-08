import xml.etree.ElementTree as ET
import pandas as pd

def parse_manifest():
    # Parse the XML file
    tree = ET.parse('C_training_data/manifest.xml')
    root = tree.getroot()
    
    # Create lists to store the data
    data = []
    
    # Extract information from each testcase
    for testcase in root.findall('.//testcase'):
        file_elem = testcase.find('file')
        if file_elem is not None:
            flaw_elem = file_elem.find('flaw')
            if flaw_elem is not None:
                try:
                    data.append({
                        'file_path': file_elem.get('path', ''),
                        'flaw_line': flaw_elem.get('line', ''),
                        'flaw_type': flaw_elem.get('name', '')
                    })
                except AttributeError:
                    continue
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    return df

# Example usage
if __name__ == "__main__":
    df = parse_manifest()
    print("Dataset Overview:")
    print(f"Total vulnerabilities: {len(df)}")
    print("\nVulnerability types:")
    print(df['flaw_type'].value_counts())
    
    # Save to CSV for easier analysis
    df.to_csv('vulnerability_dataset.csv', index=False)