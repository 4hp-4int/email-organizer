import time

from loguru import logger
from msgraph import GraphRequestAdapter


class LoggingRequestAdapter(GraphRequestAdapter):
    def __init__(self, auth_provider, **kwargs):
        super().__init__(auth_provider=auth_provider, **kwargs)

    def send(self, request, **kwargs):
        logger.debug("Sending Request: %s %s", request.method, request.url)
        if request.headers:
            logger.debug("Request Headers: %s", request.headers)
        if hasattr(request, "body") and request.body:
            logger.debug("Request Body: %s", request.body)

        start_time = time.time()
        response = super().send(request, **kwargs)
        elapsed = time.time() - start_time

        logger.debug(
            "Response: %s (Elapsed: %.2f seconds)", response.status_code, elapsed
        )
        return response
