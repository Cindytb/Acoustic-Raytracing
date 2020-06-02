#include "corelink.h"

CorelinkWrapper::RecvStream receiver_stream;
CorelinkWrapper::SendStream sender_stream;
osc::OptixSetup *setup;
float* output;
void receive_callback(const char *source, const char *msg, int size, long long timestamp)
{
  std::cout << "Received Data from " << source << " at " << timestamp << std::endl;
  float *input = (float *)msg;
  setup->get_microphones()[0]->attach_output(output);
  setup->get_sources()[0]->add_buffer(input);

  const char *char_output = reinterpret_cast<const char *>(output);
  std::string string_output(char_output, char_output + FRAMES_PER_BUFFER * 2 * sizeof(float));
  sender_stream.Send(string_output);
}

void signal_callback_handler(int signum)
{
  std::cout << "Cleaning up corelink" << std::endl;
  CorelinkWrapper::CorelinkCleanup();
  std::cout << "Done cleaning corelink" << std::endl;
  // Terminate program
  exit(signum);
}

bool Connect()
{

  signal(SIGABRT, signal_callback_handler);
  signal(SIGINT, signal_callback_handler);
  signal(SIGTERM, signal_callback_handler);
  CorelinkWrapper::CorelinkInit();
  try
  {
    //CorelinkWrapper::setServerAddress("127.0.0.1", 20010);
    CorelinkWrapper::setServerAddress("128.122.215.23", 20010);
    CorelinkWrapper::setServerCredentials("Testuser3", "Testpassword");
    CorelinkWrapper::ServerConnect();
  }
  catch (CorelinkWrapper::ErrorMsg error)
  {
    std::cout << error << std::endl;
    return false;
  }

  std::string token = CorelinkWrapper::getToken();
  std::cout << token << std::endl;

  return true;
}
int corelink_loop(osc::OptixSetup *renderer)
{
  output = new float[FRAMES_PER_BUFFER];
  setup = renderer;
  try
  {
    Connect();
    std::string meta("Dummy metadata");
    std::cout << "\n----Creating Sender:" << std::endl;
    sender_stream = CorelinkWrapper::addSender("Holodeck", "audio", meta, true, true, (int)CorelinkWrapper::STATE_UDP);
    std::cout << CorelinkWrapper::StreamData::getSenderData(sender_stream) << std::endl;

    std::cout << "\n----Creating Receiver:" << std::endl;
    receiver_stream = CorelinkWrapper::addReceiver("Holodeck", {"audio", "receiver", "infinite"}, meta, false, true, (int)CorelinkWrapper::STATE_UDP);
    receiver_stream.SetOnRecieve(&receive_callback);
    std::cout << CorelinkWrapper::StreamData::getReceiverData(receiver_stream) << std::endl;

    std::cout << "Receiver streams must have type 'reverb_input' and this client will send a stream of type 'reverb_output'" << std::endl;

    std::cout << "Hit 'q' and 'Enter' to quit the program" << std::endl;
    char q = 'a';
    while (q != 'q')
    {
      q = getchar();
    }
  }
  catch (CorelinkWrapper::ErrorMsg error)
  {
    std::cout << "CORELINK ERROR" << std::endl;
    std::cout << error << std::endl;
  }
  catch (const char *msg)
  {
    std::cout << "MISC ERROR" << std::endl;
    std::cout << msg << std::endl;
  }
  catch (...)
  {
    std::cout << "OTHER ERROR" << std::endl;
  }
  std::cout << "Cleaning up corelink" << std::endl;
  CorelinkWrapper::CorelinkCleanup();
}