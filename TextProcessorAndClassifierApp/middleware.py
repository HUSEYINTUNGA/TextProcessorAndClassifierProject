class DatasetMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.shared_data = {
            'app_dataset': None,
            'user_csv_file': None,
            'target_column': None
        }
        response = self.get_response(request)
        return response
