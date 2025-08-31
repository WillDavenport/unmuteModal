export const useBackendServerUrl = () => {
  const useProdUrl = process.env.NEXT_PUBLIC_USE_PROD_URL === "true";
  const backendServerUrl = useProdUrl 
    ? "https://willdavenport--voice-stack-orchestratorservice-web.modal.run"
    : "https://willdavenport--voice-stack-orchestratorservice-web-dev.modal.run";

  // // Get the backend server URL. This is a bit involved to support different deployment methods.
  // useEffect(() => {
  //   if (typeof window !== "undefined") {
  //     const isInDocker = ["true", "1"].includes(process.env.NEXT_PUBLIC_IN_DOCKER?.toLowerCase() || "");

  //     const prefix = isInDocker ? "/api" : "";

  //     const backendUrl = new URL("", window.location.href);
  //     if (!isInDocker) {
  //       backendUrl.port = "8000";
  //     }
  //     backendUrl.pathname = prefix;
  //     backendUrl.search = ""; // strip any query parameters
  //     setBackendServerUrl(backendUrl.toString().replace(/\/$/, "")); // remove trailing slash
  //   }
  // }, []);

  return backendServerUrl;
};
