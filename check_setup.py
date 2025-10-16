"""
Environment Setup Checker
Run this script to verify your system is ready to run the PDF RAG application
"""
import os
import sys


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.10+")
        return False


def check_env_file():
    """Check .env file exists and has required keys"""
    print("\n🔑 Checking .env file...")
    
    if not os.path.exists(".env"):
        print("   ❌ .env file not found")
        print("   📝 Create a .env file with your API keys")
        return False
    
    with open(".env", "r") as f:
        content = f.read()
    
    has_google_key = "GOOGLE_API_KEY" in content and "your_google_api_key_here" not in content
    has_hf_key = "HF_API_KEY" in content and "your_huggingface_api_key_here" not in content
    
    if has_google_key:
        print("   ✅ GOOGLE_API_KEY - Found")
    else:
        print("   ❌ GOOGLE_API_KEY - Missing or not configured")
    
    if has_hf_key:
        print("   ✅ HF_API_KEY - Found")
    else:
        print("   ❌ HF_API_KEY - Missing or not configured")
    
    return has_google_key and has_hf_key


def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "streamlit",
        "docling",
        "chromadb",
        "google.generativeai",
        "dotenv",
        "fitz",  # PyMuPDF
        "PIL",   # Pillow
        "requests"
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            if package == "google.generativeai":
                __import__("google.generativeai")
            elif package == "dotenv":
                __import__("dotenv")
            elif package == "fitz":
                __import__("fitz")
            elif package == "PIL":
                __import__("PIL")
            else:
                __import__(package)
            print(f"   ✅ {package} - Installed")
        except ImportError:
            print(f"   ❌ {package} - Not installed")
            all_installed = False
    
    if not all_installed:
        print("\n   💡 Run: pip install -r requirements.txt")
    
    return all_installed


def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")
    
    required_dirs = ["extracted_images", "chroma_db", "utils"]
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ {dir_name}/ - Exists")
        else:
            print(f"   ❌ {dir_name}/ - Missing")
            all_exist = False
    
    if not all_exist:
        print("\n   💡 Create missing directories:")
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                print(f"      mkdir {dir_name}")
    
    return all_exist


def check_utils_files():
    """Check if utility modules exist"""
    print("\n🔧 Checking utility modules...")
    
    required_files = [
        "utils/__init__.py",
        "utils/pdf_processor.py",
        "utils/image_describer.py",
        "utils/vector_store.py",
        "utils/rag_engine.py"
    ]
    
    all_exist = True
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"   ✅ {file_name} - Exists")
        else:
            print(f"   ❌ {file_name} - Missing")
            all_exist = False
    
    return all_exist


def main():
    """Main check function"""
    print("=" * 60)
    print("    PDF RAG System - Environment Setup Checker")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_env_file(),
        check_dependencies(),
        check_directories(),
        check_utils_files()
    ]
    
    print("\n" + "=" * 60)
    
    if all(checks):
        print("🎉 SUCCESS! Your environment is ready!")
        print("\n   Run the application with:")
        print("   streamlit run app.py")
    else:
        print("⚠️  SETUP INCOMPLETE - Please fix the issues above")
        print("\n   See SETUP.md for detailed instructions")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

