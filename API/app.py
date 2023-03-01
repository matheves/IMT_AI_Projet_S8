from flask import Flask, jsonify, request, Response
import logging
import os
import shutil
import subprocess
import time
import json
import psutil
import signal

app = Flask(__name__)


#Configure logger
logger = logging.getLogger('api')
logger.setLevel(logging.DEBUG)

# Create a file handler and add it to the logger
fh = logging.FileHandler('api.log')
fh.setLevel(logging.DEBUG)

# Create a console handler and add it to the logger
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

@app.route('/hello')
def hello():
    return 'Hello, World!'

@app.route('/cleanup', methods=['POST'])
def cleanup():

    id = request.json.get('session_id')
    model = request.json.get('model')

    folder_path = os.path.join('..', model, id)
    if os.path.exists(folder_path):
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
        logger.info(f'Deleted folder {id}')
        return f'Deleted folder {id}'
    else:
        logger.error(f'Folder {id} does not exist')
        return f'Folder {id} does not exist'

@app.route('/run', methods=['POST'])
def run_model():
    logger.info("Getting data from request")
    session_id = request.json.get('session_id')
    model_name = request.json.get('model_name')
    params = dict(request.json.get('params'))
    logger.info("Data stored in variables")
    script_path = '{model_name}.py'
    logger.info("Script path: " + script_path)
    
    logger.info("Params: {params}")
    # Add the session ID to the parameters
    params['run_session_id'] = session_id
        
    # Construct the command
    cmd_params = ' '.join([f"--{k} {v}" for k, v in params.items()])
    command = f"python {model_name}.py {cmd_params}"
        
    # Start the process and return the PID
    logger.info(f"Starting process")
    proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    time.sleep(1)  # Wait for 1 second
    logger.info("Getting process status")
    status_code = proc.poll()
    logger.info("Sending response")
    pid = 0
    if status_code == None:

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if session_id in proc.info['cmdline']:
                pid = proc.info['pid']

        logger.info("Process completed successfully, PID: " + str(pid) + " Status: " + str(status_code))
        return jsonify({'pid': pid, 'status': 'success'})
    else:
        logger.error("Process failed, PID: " + str(pid) + " Status: " + str(status_code))
        return jsonify({'pid': pid, 'status': 'failure', 'return_code': status_code})
    
@app.route('/stop', methods=['POST'])
def stop_model():
    pid = request.json.get('pid')

    # Check if process is running
    if pid in [p.pid for p in psutil.process_iter()]:
        # Kill the process
        os.kill(pid, signal.SIGKILL)
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'failure'})

@app.route('/<model_name>/<session_id>/data', methods=['GET'])
def get_data(model_name, session_id):

    log_file_path = f"{model_name}/{session_id}/log.txt"
    data = []
    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                step, data_name, value = line.split(';')
                data.append({'step': int(step), 'label': data_name, 'value': float(value)})
    return Response(json.dumps(data), mimetype='application/json')

@app.route('/<model>/<session_id>/status', methods=['GET'])
def get_status(session_id):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if session_id in proc.info['cmdline']:
            return jsonify({'status': 'running'})
    return jsonify({'status': 'stopped'})
    
if __name__ == '__main__':
    app.run(port=5050, host='0.0.0.0')

