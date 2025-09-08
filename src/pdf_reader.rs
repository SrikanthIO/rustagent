use std::fs;
use std::path::Path;

pub fn extract_text_from_pdf(file_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = fs::read(file_path)?;
    let doc = pdf_extract::extract_text_from_mem(&bytes)?;
    Ok(doc)
}

pub fn is_pdf_file(file_path: &Path) -> bool {
    if let Some(extension) = file_path.extension() {
        extension.to_string_lossy().to_lowercase() == "pdf"
    } else {
        false
    }
}

pub fn read_document_content(file_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    if is_pdf_file(file_path) {
        println!("Reading PDF: {}", file_path.display());
        extract_text_from_pdf(file_path)
    } else {
        println!("Reading text file: {}", file_path.display());
        fs::read_to_string(file_path).map_err(|e| e.into())
    }
}
