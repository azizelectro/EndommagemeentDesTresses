# Streamlit Cloud Deployment Guide

## Files Required for Deployment

Make sure you have these files in your repository:

1. `app.py` - Main Streamlit application
2. `detector.py` - Leak detection class
3. `requirements.txt` - Python dependencies
4. `packages.txt` - System dependencies (for OpenCV)
5. `best.pt` - Your trained YOLO model
6. `ocp-seeklogo.svg` - Logo file

## Deployment Steps

1. **Push to GitHub**: Make sure all files are committed and pushed to your GitHub repository.

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set the main file path to `app.py`
   - Click "Deploy"

## Troubleshooting OpenCV Issues

If you encounter OpenCV import errors:

1. **Check packages.txt**: Make sure the `packages.txt` file is in your repository root
2. **Verify requirements.txt**: Ensure all dependencies are properly specified
3. **Model file**: Make sure `best.pt` is in the repository root
4. **Logo file**: Ensure `ocp-seeklogo.svg` is present

## Common Issues and Solutions

### OpenCV Import Error
- The `packages.txt` file should resolve most OpenCV system dependency issues
- If still failing, try updating to a different OpenCV version in requirements.txt

### Model Loading Error
- Ensure `best.pt` is in the repository root
- Check that the model file is not corrupted
- Verify the model was trained with a compatible YOLO version

### Memory Issues
- Streamlit Cloud has memory limits
- Consider using a smaller model or optimizing the code
- Remove unnecessary imports and variables

## Testing Locally

Before deploying, test locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables

If you need to set environment variables in Streamlit Cloud:
- Go to your app settings
- Add environment variables in the "Secrets" section
- Access them in your code with `st.secrets["variable_name"]`
