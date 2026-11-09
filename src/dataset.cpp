#include "dt/dataset.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "dt/common.hpp"

namespace dt {

namespace {

constexpr char kMagic[8] = {'D', 'T', 'R', 'E', 'E', 'B', 'I', 'N'};
constexpr std::uint64_t kHeaderBytes = 24;

void seek(std::ifstream& in, std::uint64_t offset, const std::string& path) {
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!in) throw std::runtime_error("seek failed in " + path);
}

}  // namespace

DataHeader read_header(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);

  char magic[8];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
    throw std::runtime_error(path + " is not a DTREEBIN file");
  }

  DataHeader h;
  in.read(reinterpret_cast<char*>(&h.n_rows), sizeof(h.n_rows));
  in.read(reinterpret_cast<char*>(&h.n_features), sizeof(h.n_features));
  in.read(reinterpret_cast<char*>(&h.n_classes), sizeof(h.n_classes));
  if (!in) throw std::runtime_error("truncated header in " + path);
  if (h.n_rows == 0 || h.n_features == 0) throw std::runtime_error("empty dataset " + path);
  if (h.n_classes < 2 || h.n_classes > static_cast<std::uint32_t>(kMaxClasses)) {
    throw std::runtime_error("unsupported number of classes in " + path);
  }
  return h;
}

RawBlock read_block(const std::string& path, const DataHeader& header,
                    std::uint64_t row_begin, std::uint64_t row_end,
                    std::uint32_t feat_begin, std::uint32_t feat_end) {
  if (row_begin > row_end || row_end > header.n_rows || feat_begin > feat_end ||
      feat_end > header.n_features) {
    throw std::runtime_error("invalid block requested from " + path);
  }

  RawBlock block;
  block.row_begin = row_begin;
  block.n_rows = row_end - row_begin;
  block.feat_begin = feat_begin;
  block.n_features = feat_end - feat_begin;
  block.labels.resize(block.n_rows);
  block.values.resize(static_cast<std::size_t>(block.n_features) * block.n_rows);

  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);

  seek(in, kHeaderBytes + row_begin, path);
  in.read(reinterpret_cast<char*>(block.labels.data()),
          static_cast<std::streamsize>(block.n_rows));

  const std::uint64_t values_offset = kHeaderBytes + header.n_rows;
  for (std::uint32_t f = feat_begin; f < feat_end; ++f) {
    const std::uint64_t offset =
        values_offset + (static_cast<std::uint64_t>(f) * header.n_rows + row_begin) * sizeof(float);
    seek(in, offset, path);
    in.read(reinterpret_cast<char*>(block.values.data() + (f - feat_begin) * block.n_rows),
            static_cast<std::streamsize>(block.n_rows * sizeof(float)));
  }
  if (!in) throw std::runtime_error("read failed in " + path);
  return block;
}

}  // namespace dt
