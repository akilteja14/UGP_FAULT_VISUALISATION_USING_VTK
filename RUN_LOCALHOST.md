# Run on Local Host
First login in into the remote host using ssh
we are using 172.21.21.158 (Remote Server IP Address)

do the process on 2 terminals simultaneously

Run the backend_server.py in 1st terminal:

```bash
cd UGP_FAULT_VISUALISATION_USING_VTK/backend
python backend_server.py
```

Run the main.py in 2nd terminal:

```bash
cd UGP_FAULT_VISUALISATION_USING_VTK/frontend
python main.py --server
```

The user/client can open: http://172.27.21.158:8081 in google and use this website
