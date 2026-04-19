# OncoContext MCP Server

## Project Description
OncoContext is an Oncology Literature Search and Lab Data Cross-Reference MCP (Multi-Contextual Protocol) Server designed to provide robust tools for researchers and clinicians. Leveraging advanced AI capabilities, it enables seamless access to oncology literature and aids in the analysis of lab data for informed decision-making.

## Features
- **Literature Search**: Quickly access extensive research articles and papers in oncology.
- **Lab Data Cross-Referencing**: Analyzes and cross-references lab data with literature to derive actionable insights.
- **User-Friendly Interface**: Intuitive design making it easy for users to navigate and utilize the server's functionality.
- **Modular Architecture**: Fully extensible with plans for future feature enhancements.

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/BruceWayne4/onco-test.git
   cd onco-test
   ```
2. Ensure Python 3.11+ is installed.
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples
- Start the server:
   ```bash
   python app.py
   ```
- Perform a literature search:
   ```bash
   python search.py --query "lung cancer"
   ```

## Project Structure
```plaintext
/onco-test
│
├── app.py           # Main application file
├── requirements.txt  # List of dependencies
├── search.py         # Script for literature searches
└── README.md         # Project documentation
```

## Dependencies
- Python 3.11+
- FastMCP
- ChromaDB
- sentence-transformers
- [Additional dependencies as specified in requirements.txt]

## Development Setup
- Ensure that you have the necessary tools and libraries (as per dependencies).
- Follow installation instructions to set up the project locally.

## Contributing Guidelines
1. **Fork the repository.**
2. **Create your feature branch:** `git checkout -b feature-Name`
3. **Commit your changes:** `git commit -m 'Add some feature'`
4. **Push to the branch:** `git push origin feature-Name`
5. **Open a Pull Request.**

### Code of Conduct
Please adhere to the [Contributor Covenant](https://www.contributor-covenant.org/) when participating in this project.