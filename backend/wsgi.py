from app import app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Gunicorn WSGI on port {port}")
    app.run(host="0.0.0.0", port=port)
