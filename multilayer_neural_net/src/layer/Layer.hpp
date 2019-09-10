template <class Input, class Output>
class Layer {
public:
    explicit Layer() = default;
    virtual ~Layer() = default;

    virtual Output infer(const Input& input) = 0;
};
