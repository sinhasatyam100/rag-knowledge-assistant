#!/bin/bash
python3 -c "
import http.server, threading, os
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'worker healthy')
    def log_message(self, *args): pass
server = http.server.HTTPServer(('0.0.0.0', int(os.environ.get('PORT', 8080))), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print('Health server started on port', os.environ.get('PORT', 8080))
" &
exec celery -A tasks worker --loglevel=info --concurrency=2