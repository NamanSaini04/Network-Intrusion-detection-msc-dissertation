# Deploy the Streamlit dashboard

The dashboard entrypoint is:

```text
streamlit_app.py
```

## Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Click **Create app**.
3. Choose the repository:
   `NamanSaini04/Network-Intrusion-detection-msc-dissertation`
4. Branch: `main`
5. App file: `streamlit_app.py`
6. Deploy.

The repository is public and already includes `requirements.txt`.

After deployment, add the resulting `.streamlit.app` URL to the GitHub repository description, your CV, LinkedIn and portfolio website.

## Local run

```bash
git clone https://github.com/NamanSaini04/Network-Intrusion-detection-msc-dissertation.git
cd Network-Intrusion-detection-msc-dissertation
pip install -r requirements.txt
streamlit run streamlit_app.py
```
