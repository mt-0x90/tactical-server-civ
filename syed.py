from flask import Flask, request, jsonify
import socket

app = Flask(__name__)

def send_udp(message):
    server_address = (sserver_ip, server_port)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
        client_socket.sendto(message.encode(), server_address)
        # Receive response from the server
    client_socket.close()

# Define the /udp endpoint
@app.route('/udp', methods=['POST'])
def udp_endpoint():
    # Check if the request contains JSON data
    if request.is_json:
        # Parse the JSON data
        data = request.get_json()
        udp_value = data.get("udp", None)
        #send_udp(udp_value)
        # Check if "udp" key exists and matches "1"
        if udp_value == "1":
            response = {
                "status": "success",
                "message": "Received valid UDP data",
                "udp_value": udp_value
            }
            return jsonify(response), 200
        else:
            return jsonify({"status": "error", "message": "Invalid value for 'udp'"}), 400
    else:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
